import os
import csv
import tqdm
import httpx
import dotenv
import asyncio
import logging
import argparse
import pandas as pd

harm_categories = [
    'HARM_CATEGORY_HATE_SPEECH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'HARM_CATEGORY_DANGEROUS_CONTENT',
    'HARM_CATEGORY_HARASSMENT',
]


async def make_request(session, api_key, text):
    url = (
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key='
        + api_key
    )
    headers = {'Content-Type': 'application/json'}
    payload = {
        'safetySettings': [
            {
                'category': cat,
                'threshold': 'BLOCK_NONE',
            }
            for cat in harm_categories
        ],
        'contents': [{'parts': [{'text': text}]}],
    }

    try:
        response = await session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        logging.error('Request timed out.')
    except httpx.HTTPStatusError as e:
        logging.error(f'HTTP error occurred: {e}')
        await asyncio.sleep(10)
    except Exception as e:
        logging.error(f'An error occurred: {e}')


def convert_text_to_prompt(text):
    return f"""You are an expert english to hungarian translator. You
    translate chat messages from english to hungarian, while keeping the
    original meaning and style. Keep shortened words and references to
    english entities as they are.
    ### Example
    English: Antitrust enforcement agencies like the FTC or DOJ could investigate employers that are abusing their market power.
    Hungarian: Az FTC vagy a DOJ nevű antitröszt-hatóságok vizsgálhatják azokat a munkáltatókat, akik visszaélnek a piaci hatalmukkal.
    ### Task
    English: {text}
    Hungarian:"""


def checkpoint_response_csv(row, response_text):
    # if the file does not exist, create it and write the header
    if not os.path.exists('data/oasst1-en-hu.csv'):
        with open('data/oasst1-en-hu.csv', 'w') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(list(row.index) + ['hungarian_translation'])
    with open('data/oasst1-en-hu.csv', 'a') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow([*row.values, response_text])


async def bundle_row_and_request(row, session, api_key, prompt):
    req = await make_request(session, api_key, prompt)
    if req is not None:
        print(req)
        try:
            if 'candidates' not in req:
                reason = req.get('promptFeedback', None)
                if reason is not None:
                    reason = reason.get('blockReason', None)
                return (row, f'BLOCKED: {reason}')
            hungarian_translation = req['candidates'][0]['content']['parts'][
                0
            ]['text']

            return (row, hungarian_translation)
        except Exception as e:
            logging.error(f'An error occurred: {e}', exc_info=True)
            return (row, None)
    return (row, req)


async def main(api_key, continue_from=None):
    df = pd.read_csv('data/oasst1-en.csv', quoting=csv.QUOTE_NONNUMERIC)
    if continue_from is not None:
        last_idx = df.query(f'message_id == "{continue_from}"').index
        df = df.iloc[last_idx[0] :]
    async with httpx.AsyncClient(timeout=15) as session:
        in_flight = []
        for _, row in tqdm.tqdm(
            df.iterrows(),
            total=len(df),
            desc='Translating',
            unit='message',
        ):
            text = row['text']
            prompt = convert_text_to_prompt(text)
            in_flight.append(
                bundle_row_and_request(row, session, api_key, prompt)
            )
            if len(in_flight) == 5:
                responses = await asyncio.gather(*in_flight)
                await asyncio.sleep(0.2)
                failed_count = len(
                    [response for response in responses if response[1] is None]
                )
                if failed_count == 5:
                    logging.error('5 failed requests in a row. Exiting.')
                    break
                for row, translation in responses:
                    checkpoint_response_csv(row, translation)
                in_flight = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue',
        dest='continue_',
        action='store_true',
        help='Continue from where the last translation left off.',
        default=False,
    )
    args = parser.parse_args()
    last_message_id = None
    if args.continue_:
        checkpoint = pd.read_csv('data/oasst1-en-hu.csv')
        last_message_id = checkpoint.tail(1)['message_id'].item()

    dotenv.load_dotenv()
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(api_key, last_message_id))
