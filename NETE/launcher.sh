#!/bin/bash

#SBATCH --job-name="NETE"

#SBATCH --qos=debug

#SBATCH -D .

#SBATCH --output=%x_%j.out

#SBATCH --error=%x_%j.err

#SBATCH --cpus-per-task=40

#SBATCH --gres=gpu:1

#SBATCH --time=02:00:00

#SBATCH --cpu-freq=2000000

module purge; module load gcc/8.3.0 ffmpeg/4.2.1 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML torch/1.9.0a0

###################### TripAdvisor #############################
python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/TripAdvisor/reviews.pickle \
-id ./Data/TripAdvisor/1/ \
-pp ./Result/TripAdvisor/NETE/1/ >> ./Result/TripAdvisor/NETE/TP_NETE_1.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/TripAdvisor/reviews.pickle \
-id ./Data/TripAdvisor/2/ \
-pp ./Result/TripAdvisor/NETE/2/ >> ./Result/TripAdvisor/NETE/TP_NETE_2.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/TripAdvisor/reviews.pickle \
-id ./Data/TripAdvisor/3/ \
-pp ./Result/TripAdvisor/NETE/3/ >> ./Result/TripAdvisor/NETE/TP_NETE_3.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/TripAdvisor/reviews.pickle \
-id ./Data/TripAdvisor/4/ \
-pp ./Result/TripAdvisor/NETE/4/ >> ./Result/TripAdvisor/NETE/TP_NETE_4.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/TripAdvisor/reviews.pickle \
-id ./Data/TripAdvisor/5/ \
-pp ./Result/TripAdvisor/NETE/5/ >> ./Result/TripAdvisor/NETE/TP_NETE_5.log

###################### Yelp #############################
# python -u NETE/run.py -gd 0 -pf 0 \
# -dp ./Data/Yelp/reviews.pickle \
# -id ./Data/Yelp/1/ \
# -pp ./Result/Yelp/NETE/1/ >> ./Result/Yelp/NETE/Yelp_NETE_1.log

# python -u NETE/run.py -gd 0 -pf 0 \
# -dp ./Data/Yelp/reviews.pickle \
# -id ./Data/Yelp/2/ \
# -pp ./Result/Yelp/NETE/2/ >> ./Result/Yelp/NETE/Yelp_NETE_2.log

# python -u NETE/run.py -gd 0 -pf 0 \
# -dp ./Data/Yelp/reviews.pickle \
# -id ./Data/Yelp/3/ \
# -pp ./Result/Yelp/NETE/3/ >> ./Result/Yelp/NETE/Yelp_NETE_3.log

# python -u NETE/run.py -gd 0 -pf 0 \
# -dp ./Data/Yelp/reviews.pickle \
# -id ./Data/Yelp/4/ \
# -pp ./Result/Yelp/NETE/4/ >> ./Result/Yelp/NETE/Yelp_NETE_4.log

# python -u NETE/run.py -gd 0 -pf 0 \
# -dp ./Data/Yelp/reviews.pickle \
# -id ./Data/Yelp/5/ \
# -pp ./Result/Yelp/NETE/5/ >> ./Result/Yelp/NETE/Yelp_NETE_5.log

###################### Amazon - ClothingShoesAndJewelry  #############################
python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
-id ./Data/Amazon/ClothingShoesAndJewelry/1/ \
-pp ./Result/Amazon/ClothingShoesAndJewelry/NETE/1/ >> ./Result/Amazon/ClothingShoesAndJewelry/NETE/Amz_CSJ_NETE_1.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
-id ./Data/Amazon/ClothingShoesAndJewelry/2/ \
-pp ./Result/Amazon/ClothingShoesAndJewelry/NETE/2/ >> ./Result/Amazon/ClothingShoesAndJewelry/NETE/Amz_CSJ_NETE_2.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
-id ./Data/Amazon/ClothingShoesAndJewelry/3/ \
-pp ./Result/Amazon/ClothingShoesAndJewelry/NETE/3/ >> ./Result/Amazon/ClothingShoesAndJewelry/NETE/Amz_CSJ_NETE_3.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
-id ./Data/Amazon/ClothingShoesAndJewelry/4/ \
-pp ./Result/Amazon/ClothingShoesAndJewelry/NETE/4/ >> ./Result/Amazon/ClothingShoesAndJewelry/NETE/Amz_CSJ_NETE_4.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
-id ./Data/Amazon/ClothingShoesAndJewelry/5/ \
-pp ./Result/Amazon/ClothingShoesAndJewelry/NETE/5/ >> ./Result/Amazon/ClothingShoesAndJewelry/NETE/Amz_CSJ_NETE_5.log

###################### Amazon - MoviesAndTV  #############################
python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/MoviesAndTV/reviews.pickle \
-id ./Data/Amazon/MoviesAndTV/1/ \
-pp ./Result/Amazon/MoviesAndTV/NETE/1/ >> ./Result/Amazon/MoviesAndTV/NETE/Amz_MT_NETE_1.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/MoviesAndTV/reviews.pickle \
-id ./Data/Amazon/MoviesAndTV/2/ \
-pp ./Result/Amazon/MoviesAndTV/NETE/2/ >> ./Result/Amazon/MoviesAndTV/NETE/Amz_MT_NETE_2.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/MoviesAndTV/reviews.pickle \
-id ./Data/Amazon/MoviesAndTV/3/ \
-pp ./Result/Amazon/MoviesAndTV/NETE/3/ >> ./Result/Amazon/MoviesAndTV/NETE/Amz_MT_NETE_3.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/MoviesAndTV/reviews.pickle \
-id ./Data/Amazon/MoviesAndTV/4/ \
-pp ./Result/Amazon/MoviesAndTV/NETE/4/ >> ./Result/Amazon/MoviesAndTV/NETE/Amz_MT_NETE_4.log

python -u NETE/run.py -gd 0 -pf 0 \
-dp ./Data/Amazon/MoviesAndTV/reviews.pickle \
-id ./Data/Amazon/MoviesAndTV/5/ \
-pp ./Result/Amazon/MoviesAndTV/NETE/5/ >> ./Result/Amazon/MoviesAndTV/NETE/Amz_MT_NETE_5.log

