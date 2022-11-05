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
# python -u PETER/main.py --cuda --peter_mask \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/1/ \
# --checkpoint ./Result/TripAdvisor/PETER/1/ \
# --use_feature >> ./Result/TripAdvisor/PETER/TP_PETER_1.log

# python -u PETER/main.py --cuda --peter_mask \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/2/ \
# --checkpoint ./Result/TripAdvisor/PETER/2/ \
# --use_feature >> ./Result/TripAdvisor/PETER/TP_PETER_2.log

# python -u PETER/main.py --cuda --peter_mask \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/3/ \
# --checkpoint ./Result/TripAdvisor/PETER/3/ \
# --use_feature >> ./Result/TripAdvisor/PETER/TP_PETER_3.log

# python -u PETER/main.py --cuda --peter_mask \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/4/ \
# --checkpoint ./Result/TripAdvisor/PETER/4/ \
# --use_feature >> ./Result/TripAdvisor/PETER/TP_PETER_4.log

# python -u PETER/main.py --cuda --peter_mask \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/5/ \
# --checkpoint ./Result/TripAdvisor/PETER/5/ \
# --use_feature >> ./Result/TripAdvisor/PETER/TP_PETER_5.log

###################### Yelp #############################
# python -u PETER/main.py --cuda --peter_mask \
# --mode Test \
# --log_interval 800 \
# --data_path ./Data/Yelp/reviews.pickle \
# --index_dir ./Data/Yelp/1/ \
# --checkpoint ./Result/Yelp/PETER/1/ \
# --use_feature >> ./Result/Yelp/PETER/Yelp_usefeat_1_test.log

# python -u PETER/main.py --cuda --peter_mask \
# --mode Test \
# --log_interval 800 \
# --data_path ./Data/Yelp/reviews.pickle \
# --index_dir ./Data/Yelp/2/ \
# --checkpoint ./Result/Yelp/PETER/2/ \
# --use_feature >> ./Result/Yelp/PETER/Yelp_usefeat_2_test.log

# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Yelp/reviews.pickle \
# --index_dir ./Data/Yelp/3/ \
# --checkpoint ./Result/Yelp/PETER/3/ \
# --use_feature >> ./Result/Yelp/PETER/Yelp_usefeat_3.log

# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Yelp/reviews.pickle \
# --index_dir ./Data/Yelp/4/ \
# --checkpoint ./Result/Yelp/PETER/4/ \
# --use_feature >> ./Result/Yelp/PETER/Yelp_usefeat_4.log

# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Yelp/reviews.pickle \
# --index_dir ./Data/Yelp/5/ \
# --checkpoint ./Result/Yelp/PETER/5/ \
# --use_feature >> ./Result/Yelp/PETER/Yelp_usefeat_5.log

# ###################### Amazon - ClothingShoesAndJewelry #############################
# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/1/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PETER/1/ \
# --use_feature >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PETER_1.log

# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/2/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PETER/2/ \
# --use_feature >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PETER_2.log

# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/3/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PETER/3/ \
# --use_feature >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PETER_3.log

# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/4/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PETER/4/ \
# --use_feature >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PETER_4.log

# python -u PETER/main.py --cuda --peter_mask \
# --log_interval 800 \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/5/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/PETER/5/ \
# --use_feature >> ./Result/Amazon/ClothingShoesAndJewelry/PETER/Amz_CSJ_PETER_5.log

# ###################### Amazon - MoviesAndTV #############################

python -u PETER/main.py --cuda --peter_mask \
--log_interval 800 \
--mode Test \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/1/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PETER/1/ \
--use_feature >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PETER_1_Test.log

python -u PETER/main.py --cuda --peter_mask \
--log_interval 800 \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/2/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PETER/2/ \
--use_feature >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PETER_2.log

python -u PETER/main.py --cuda --peter_mask \
--log_interval 800 \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/3/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PETER/3/ \
--use_feature >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PETER_3.log

python -u PETER/main.py --cuda --peter_mask \
--log_interval 800 \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/4/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PETER/4/ \
--use_feature >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PETER_4.log

python -u PETER/main.py --cuda --peter_mask \
--log_interval 800 \
--data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir ./Data/Amazon/MoviesAndTV/5/ \
--checkpoint ./Result/Amazon/MoviesAndTV/PETER/5/ \
--use_feature >> ./Result/Amazon/MoviesAndTV/PETER/Amz_MT_PETER_5.log