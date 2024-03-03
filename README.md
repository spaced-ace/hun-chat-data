# Hungarian chat data

*This project is for downloading the open assistant chat dataset and translating it to hungarian*

## Basic info

The translated data is limited to the english parts of the dataset. The project uses gemini for
translation, while instructing it to carefully watch for preserving the original meaning and context.
The code expects GOOGLE_AI_API_KEY to be present in a .env file in the root of the folder.

The program allows for 5 concurrent requests to stay below the rate limit but be as fast as possible.
The code is hardly anything special, chatgpt could probably have written it in a few shots, customize
it for if you need anything.

The program turns off all safeguards that are available, so gemini wont censor the responses. Still,
6 messages were blocked for me (for "other" reasons.) I used google translate to manually translate
those.


## Download the dataset from huggingface

View the [hf repo](https://huggingface.co/datasets/jazzysnake01/oasst1-en-hun-gemini)
or download it in python:

```python
import datasets

ds = datasets.load_dataset('jazzysnake01/oasst1-en-hun-gemini')
```

## Usage

Translating the english part takes around 20 hours, which is 46% of the total dataset (~41k chat messages)

Installation:
```bash
pip install -r requirements.txt
```
Usage:
```bash
pyhton acquire_data.py && python translate_data.py --timeout 15
```

Once started the code may stop if 5 requests have failed in a row, in that case you can
continue the translation from where you left off by:

```bash
pyhton translate_data.py --continue
```
If (when) the translation has finished with failed requests, the following command can
be used to patch up those mistakes. For me, 1.2k failed requests remained, but that is
because I set a 15sec timeout so long messages don't hold up 4 others with them (higher throughput).
In this case whatever timeout you use will be doubled. (--timeout flag)
```bash
pyhton translate_data.py --patch-failed
```
