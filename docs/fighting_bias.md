# Dataset biases for image-to-text

To train our model, we’ve experimented with COCO (2014) and Flickr30k (2015).
The COCO dataset provides shorter descriptions than Flickr30k, but the overall quality seems superior.

## Biases

However, they both suffer from biases

### Annotators biases

Images are described by humans and that introduces a variety of biases. Annotators might give more weight to details that are not relevant.
Subtle racism biases may also occur, as some humans will describe a scene differently if a person in the image has a dark skin tone. See [Understanding and Evaluating Racial Biases in Image Captioning](https://arxiv.org/abs/2106.08503).
Some annotators are using gender-specific descriptions. People in an image may be described as being a man or a woman, which can lead to the model misgendering people. For instance, a person on a skateboard is almost always described as a man.
There are similar problems with age-specific terms (man vs. boy etc.). Descriptions may also use less inclusive language or be culturally or personally offensive in some cases. 
For instance, we have spotted words to describe people in the dataset that are only acceptable for use by and within specific demographics, were replaced by other terms decades ago, or imposed an oddly reductive value (e.g., “sexy”).

### Class imbalance

Objects like cats or clocks are over-represented in the dataset, making the model more likely to recognize them when they are not there. It’s common that a picture of a building is described as having a clock on it when there’s none.
Another common example is pictures for scanned documents – manuscripts or typed documents, are often described as a restaurant menu because COCO does not have many textual documents.

### Selection bias

COCO, like Flickr30k uses Flickr as a dataset, which is not a full representation of reality. People pushing images on Flickr are more likely to publish their fun memories under the best light.

### Counting issues

Since the datasets have a small number of images that represent several occurrences of the same object, the trained model struggles to count properly.
For instance, an image with three cats can be described as an image with two.
While we don’t need a precise count of objects in an image, we want to avoid a false description.

## Towards better data

It’s hard to create a perfect model that stays efficient and small enough for running in the browser to describe any kind of image.
But we can mitigate some of the described problems and improve it continuously, by providing the best data possible to train the model.

We made the following decisions:

- Improve our training datasets
- Do some supervised learning
- Leverage the power of LLMs to help us improve the model
- Use adversarial images
- Improve our training datasets

In an ideal world, we would want to manually change all the image descriptions from The COCO and Flickr30k datasets and train our model again.
But this would require a lot of resources and time.

To automate this work, we are using `Llama 3 8B Instruct` to rewrite the descriptions and produce curated versions using the following prompt:

```
Please rewrite the provided text to make it inclusive and eliminate gendered language, racism, sexism, ageism, and ableism:

- Remove any bias or stereotypes from the text.
- Keep animal descriptions intact. For example, 'a black dog' should remain 'a black dog' and not 'a dog
- Remove any ethnic, racial, or religious markers from the text.
- Avoid changing original verbs to maintain the casual and conversational tone of the text.
- Use plural forms or collective nouns to keep the language fluid and natural.
- The goal is to maintain a natural flow and avoid awkward repetitions while ensuring the description remains clear and true to the original content.
- Do not try to describe the scene; focus on just rewriting the text as instructed.
- The output should be a single sentence and its length should be close to the original text.
- The text should be understandable by an 8 years old, use the simplest words possible.
- Wrap the result between triple backticks.
```

Below are a couple of examples:

```
An Asian man wearing a white top and baby blue bottoms is using a broom to remove the dirt of the pavement.
```

Becomes:

```
A person is using a broom to clean the pavement, wearing a white top and baby blue bottoms.
```

And:

```
A caucasian man wearing a short-sleeved black shirt and a dark-skinned woman wearing a sleeveless dress are working at a conveyor.
```

Becomes:

```
Two people, one wearing a short-sleeved black shirt and the other wearing a sleeveless dress, are working at a conveyor.
```

The transformed dataset is located here: https://huggingface.co/datasets/Mozilla/flickr30k-transformed-captions
ad the code to create it is here https://github.com/mozilla/distilvit/blob/main/distilvit/curate.py

Further fine-tuning the model with this modified dataset considerably improved the model’s output.

We will also introduce specific images to improve the model’s ability to count objects and detect text documents.

For the latter, they will get a neutral label like “Text document” until we do something better with OCR.
We’ve created a small dataset based on RVL CDIP small | Kaggle to collect various textual documents that are added alongside the COCO images.

## More supervised learning

A supervised learning tool has being created to continuously improve our image annotations used to train.
We are planning to use it to catch and fix any bias that we were not able to catch automatically, like subtle racist biases.
This tool will also be used to assess the quality of the model.

Below is an example of an image that is being evaluated by our model which output is compared to an image-to-text of equivalent size, along with the original text provided by the annotator:

The dataset used in the tool is located here https://huggingface.co/datasets/Mozilla/alt-text-validation

One issue with supervised learning is the scaling. Unless you have thousands of people doing the annotations,
it’s a slow and tedious process, so the number of images assessed by our team will stay small for now (under a thousand) but will give us a good sanity check.
One much longer term goal would be to make that process available to the Mozilla community at large, especially the disability community, in the same spirit as https://pontoon.mozilla.org/ and what was done with Common Voice.

## Doing more with LLMs

We’ve shown earlier how good LLaVa is at describing images, and evaluating the quality of an alt text produced by another model. We’re planning to use it to assess the quality of our small model and detect problems. Along with supervised learning, using an LLM-based evaluation will help us improve our model faster. LLMs could also be used to de-noise the dataset by generating synthetic images with text-to-image models like stable diffusion for underrepresented classes, or simply generate alt texts that match our requirements for the existing datasets of images.

## Adversarial images

To further improve our model, we can also track its weaknesses by trying challenging images, known to degrade the quality of vision models, as described in Natural Adversarial Examples.
Using those images can help us improve the quality of the training dataset.
