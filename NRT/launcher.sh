#!/bin/bash

#SBATCH --job-name="NRT"

#SBATCH --qos=debug

#SBATCH -D .

#SBATCH --output=%x_%j.out

#SBATCH --error=%x_%j.err

#SBATCH --cpus-per-task=40

#SBATCH --gres=gpu:1

#SBATCH --time=2:00:00

#SBATCH --cpu-freq=2000000

module purge; module load gcc/8.3.0 ffmpeg/4.2.1 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML torch/1.9.0a0

###################### Yelp #############################
# python -u NRT/main.py --cuda \
# --data_path ./Data/Yelp/reviews.pickle \
# --index_dir ./Data/Yelp/1/ \
# --mode Test \
# --checkpoint ./Result/Yelp/NRT/1/  >> ./Result/Yelp/NRT/Yelp_NRT_1_Test.log

python -u NRT/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/2/ \
--mode Test \
--checkpoint ./Result/Yelp/NRT/2/  >> ./Result/Yelp/NRT/Yelp_NRT_2_Test.log

python -u NRT/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/3/ \
--checkpoint ./Result/Yelp/NRT/3/  >> ./Result/Yelp/NRT/Yelp_NRT_3.log

python -u NRT/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/4/ \
--checkpoint ./Result/Yelp/NRT/4/  >> ./Result/Yelp/NRT/Yelp_NRT_4.log

python -u NRT/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/5/ \
--checkpoint ./Result/Yelp/NRT/5/  >> ./Result/Yelp/NRT/Yelp_NRT_5.log

###################### Amazon - ClothingShoesAndJewelry #############################
# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry//1/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry//NRT/1/  >> ./Result/Amazon/ClothingShoesAndJewelry//NRT/Amz_CSJ_NRT_1.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry//2/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry//NRT/2/  >> ./Result/Amazon/ClothingShoesAndJewelry//NRT/Amz_CSJ_NRT_2.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry//3/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry//NRT/3/  >> ./Result/Amazon/ClothingShoesAndJewelry//NRT/Amz_CSJ_NRT_3.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry//4/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry//NRT/4/  >> ./Result/Amazon/ClothingShoesAndJewelry//NRT/Amz_CSJ_NRT_4.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry//5/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry//NRT/5/  >> ./Result/Amazon/ClothingShoesAndJewelry//NRT/Amz_CSJ_NRT_5.log

###################### Amazon - MoviesAndTV #############################
# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV//1/ \
# --checkpoint ./Result/Amazon/MoviesAndTV//NRT/1/  >> ./Result/Amazon/MoviesAndTV//NRT/Amz_MT_NRT_1.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV//2/ \
# --checkpoint ./Result/Amazon/MoviesAndTV//NRT/2/  >> ./Result/Amazon/MoviesAndTV//NRT/Amz_MT_NRT_2.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV//3/ \
# --checkpoint ./Result/Amazon/MoviesAndTV//NRT/3/  >> ./Result/Amazon/MoviesAndTV//NRT/Amz_MT_NRT_3.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV//4/ \
# --checkpoint ./Result/Amazon/MoviesAndTV//NRT/4/  >> ./Result/Amazon/MoviesAndTV//NRT/Amz_MT_NRT_4.log

# python -u NRT/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV//5/ \
# --checkpoint ./Result/Amazon/MoviesAndTV//NRT/5/  >> ./Result/Amazon/MoviesAndTV//NRT/Amz_MT_NRT_5.log