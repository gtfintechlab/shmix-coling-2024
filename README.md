# Code for 'Saliency-Aware Hierarchical Mixup for Multimodal Financial Prediction'

Predicting price variations of financial instruments for risk modeling and stock trading is challenging due to the stochastic nature of the stock market. While recent advancements in the Financial AI realm have expanded the scope of data and methods they use, such as textual and audio cues from financial earnings calls, limitations exist. Most datasets are small, and show domain distribution shifts due to the nature of their source, suggesting the exploration for data augmentation for robust augmentation strategies such as Mixup. To tackle such challenges in the financial domain, we propose \texttt{SH-Mix}: Saliency-guided Hierarchical Mixup augmentation technique for multimodal financial prediction tasks. \texttt{SH-Mix} combines multi-level embedding mix up strategies based on the contribution of each modality and context subsequences. Through extensive quantitative and qualitative experiments on financial earnings and conference call datasets consisting of text and speech, we show that SH-Mix outperforms state-of-the-art methods by $3-7\%$. Additionally, we show that SH-Mix is generalizable across different modalities and models.

# How to run

To run for M3A or Earnings call dataset, use the following command with 'data_type' as 'm3a' and 'ec' respectively.
```
    python shmix_optuna.py --data <data_type> --tau <tau_value> --run_name <name_of_run> --num_epochs <number_of_epochs>
```

To run for MUStARD or CMU-MOSI dataset, use the following command with 'data_type' as 'mustard' and 'mosi' respectively.
```
    python shmix_video.py --data <data_type> --tau <tau_value> --run_name <name_of_run> --num_epochs <number_of_epochs>
```

For value of tau, use the following mapping
```
    tau = 3  : --tau 0
    tau = 7  : --tau 1
    tau = 15 : --tau 2
```

Environment: Python 3.9, Tensorflow 2.12.0
Conda environment can be created from the environment file (env.yml) provided.

# Datasets

The datasets can be downloaded from the respective links and processed as mentioned in the respective accompanying work. The datasets need to be put into 'shmix/Dataset' folder and used.

* M&A Calls : https://github.com/midas-research/m3a-acl
* Earnings Calls : https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction
* MUStARD : https://github.com/soujanyaporia/MUStARD
* CMU-MOSI : http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/