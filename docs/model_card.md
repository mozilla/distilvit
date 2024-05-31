---
tags:
  - image-to-text
  - image-captioning
license: apache-2.0
metrics:
  - rouge
datasets:
  - nlphuji/flickr30k
widget:
  - src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg
    example_title: Savanna
  - src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg
    example_title: Football Match
  - src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg
    example_title: Airport
base_model:
  - google/vit-base-patch16-224-in21k

model-index:
  - name: mozilla/distilvit
    results:
      - task:
          type: image-to-text
          name: Image To Text
        dataset:
          name: nlphuji/flickr30k
          type: nlphuji/flickr30k
        metrics:
          - name: ROUGE-1
            type: rouge
            value: 43.006
            verified: true
          - name: ROUGE-2
            type: rouge
            value: 16.9939
            verified: true
          - name: ROUGE-L
            type: rouge
            value: 38.8923
            verified: true
          - name: ROUGE-LSUM
            type: rouge
            value: 38.8877
            verified: true
          - name: loss
            type: loss
            value: 0.19939416646957397
          - name: gen_len
            type: gen_len
            value: 11.327256736227712
            verified: true
---

# distilvit

This model is a work in progress. Fine-tuned version of those base models:

- a VIT model for the image encoder: https://huggingface.co/google/vit-base-patch16-224-in21k
- a Distilled GPT-2 model for the text decoder: https://huggingface.co/distilbert/distilgpt2

This model was trained on:

- Flickr30k : https://huggingface.co/datasets/nlphuji/flickr30k
- COCO 2017: https://cocodataset.org

You can get that checkpoint using the 3083a3cef6e3c8dd90df3f088074bbe836b0f403 commit.

It was then further fine-tuned on :

- Flickr30k debiased: https://huggingface.co/datasets/Mozilla/flickr30k-transformed-captions
- DocOrNot: https://huggingface.co/datasets/Mozilla/docornot

You can find the code used to create the model here: https://github.com/mozilla/distilvit

### Framework versions

- Transformers 4.40.2
- Pytorch 2.3.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
