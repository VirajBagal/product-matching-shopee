# product-matching-shopee
This repo contains the code for product matching problem statement. Given image and textual description of different products, the task is to match similar products with each other. Here, I took a semi-supervised approach because the test set apparently contained mostly unseen images and their text descriptions.

Approach taken:
1. To simulate test set scenario, I implemented GroupKFold based on product labels.
2. Efficientnet series for image feature extraction. 
3. BERT for text feature extraction.
4. Multimodal BERT for joint (image+text) feature extraction.
5. ArcMargin Head for enhancing intra-class compactness and inter-class distance in the latent space.
6. Heavy augmentations like CutOut helped. 
7. Ensemble of 4-5 such models. 



# To train

```python
python train.py --run_name give_run_name --model_name tf_efficientnet_b4_ns --batch_size 8 --gpu 1
```

# To test

```python
python test.py --run_name give_run_name --model_name tf_efficientnet_b4_ns --batch_size 8 --gpu 1
```


