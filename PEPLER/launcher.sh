#!/bin/bash

#SBATCH --job-name="PETER"

#SBATCH --qos=debug

#SBATCH -D .

#SBATCH --output=%x_%j.out

#SBATCH --error=%x_%j.err

#SBATCH --cpus-per-task=40

#SBATCH --gres=gpu:1

#SBATCH --time=2:00:00

#SBATCH --cpu-freq=2000000

module purge; module load gcc/8.3.0 ffmpeg/4.2.1 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML torch/1.9.0a0
###################### TripAdvisor #############################

####### Continuous Prompt #######
python -u PEPLER/main.py --cuda \
--data_path ./Data/TripAdvisor/reviews.pickle \
--index_dir ./Data/TripAdvisor/1/ \
--checkpoint ./Result/TripAdvisor/PEPLER/1/ >> ./Result/TripAdvisor/PETER/TP_PEPLER_1_CP.log

####### Discrete Prompt   #######
python -u PEPLER/discrete.py --cuda \
--data_path ./Data/TripAdvisor/reviews.pickle \
--index_dir ./Data/TripAdvisor/1/ \
--checkpoint ./Result/TripAdvisor/PEPLER/1/ >> ./Result/TripAdvisor/PETER/TP_PEPLER_1_DP.log

#######      MF reg       #######
python -u PEPLER/reg.py --cuda --use_mf \
--data_path ./Data/TripAdvisor/reviews.pickle \
--index_dir ./Data/TripAdvisor/1/ \
--checkpoint ./Result/TripAdvisor/PEPLER/1/ >> ./Result/TripAdvisor/PETER/TP_PEPLER_1_MF.log

#######      MLP reg       #######
python -u PEPLER/reg.py --cuda --rating_reg 1 \
--data_path ./Data/TripAdvisor/reviews.pickle \
--index_dir ./Data/TripAdvisor/1/ \
--checkpoint ./Result/TripAdvisor/PEPLER/1/ >> ./Result/TripAdvisor/PETER/TP_PEPLER_1_MLP.log

###################### Yelp #############################
####### Continuous Prompt #######
python -u PEPLER/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/1/ \
--checkpoint ./Result/Yelp/PEPLER/1/ >> ./Result/Yelp/PETER/Yelp_PEPLER_1_CP.log

####### Discrete Prompt   #######
python -u PEPLER/discrete.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/1/ \
--checkpoint ./Result/Yelp/PEPLER/1/ >> ./Result/Yelp/PETER/Yelp_PEPLER_1_DP.log

#######      MF reg       #######
python -u PEPLER/reg.py --cuda --use_mf \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/1/ \
--checkpoint ./Result/Yelp/PEPLER/1/ >> ./Result/Yelp/PETER/Yelp_PEPLER_1_MF.log

#######      MLP reg       #######
python -u PEPLER/reg.py --cuda --rating_reg 1 \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/1/ \
--checkpoint ./Result/Yelp/PEPLER/1/ >> ./Result/Yelp/PETER/Yelp_PEPLER_1_MLP.log

###################### Amazon - ClothingShoesAndJewelry #############################
####### Continuous Prompt #######
python -u PEPLER/main.py --cuda \
--data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir ./Data/Amazon/ClothingShoesAndJewelry/1/ \
--checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PEPLER/1/ >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PEPLER_1_CP.log

####### Discrete Prompt   #######
python -u PEPLER/discrete.py --cuda \
--data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir ./Data/Amazon/ClothingShoesAndJewelry/1/ \
--checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PEPLER/1/ >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PEPLER_1_DP.log

#######      MF reg       #######
python -u PEPLER/reg.py --cuda --use_mf \
--data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir ./Data/Amazon/ClothingShoesAndJewelry/1/ \
--checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PEPLER/1/ >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PEPLER_1_MF.log

#######      MLP reg       #######
python -u PEPLER/reg.py --cuda --rating_reg 1 \
--data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir ./Data/Amazon/ClothingShoesAndJewelry/1/ \
--checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PEPLER/1/ >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PEPLER_1_MLP.log

###################### Amazon - MoviesAndTV #############################
####### Continuous Prompt #######
python -u PEPLER/main.py --cuda \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/1/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PEPLER/1/ >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PEPLER_1_CP.log

####### Discrete Prompt   #######
python -u PEPLER/discrete.py --cuda \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/1/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PEPLER/1/ >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PEPLER_1_DP.log

#######      MF reg       #######
python -u PEPLER/reg.py --cuda --use_mf \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/1/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PEPLER/1/ >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PEPLER_1_MF.log

#######      MLP reg       #######
python -u PEPLER/reg.py --cuda --rating_reg 1 \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/1/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PEPLER/1/ >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PEPLER_1_MLP.log