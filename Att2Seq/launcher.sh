#!/bin/bash

#SBATCH --job-name="Att2Seq"

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
# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/1/ \
# --checkpoint ./Result/TripAdvisor/Att2Seq/1/ >> ./Result/TripAdvisor/Att2Seq/TA_Att2Seq_1.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/2/ \
# --checkpoint ./Result/TripAdvisor/Att2Seq/2/ >> ./Result/TripAdvisor/Att2Seq/TA_Att2Seq_2.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/3/ \
# --checkpoint ./Result/TripAdvisor/Att2Seq/3/ >> ./Result/TripAdvisor/Att2Seq/TA_Att2Seq_3.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/4/ \
# --checkpoint ./Result/TripAdvisor/Att2Seq/4/ >> ./Result/TripAdvisor/Att2Seq/TA_Att2Seq_4.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/TripAdvisor/reviews.pickle \
# --index_dir ./Data/TripAdvisor/5/ \
# --checkpoint ./Result/TripAdvisor/Att2Seq/5/ >> ./Result/TripAdvisor/Att2Seq/TA_Att2Seq_5.log

###################### Yelp #############################
python -u Att2Seq/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/1/ \
--checkpoint ./Result/Yelp/Att2Seq/1/ >> ./Result/Yelp/Att2Seq/Yelp_Att2Seq_1.log

python -u Att2Seq/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/2/ \
--checkpoint ./Result/Yelp/Att2Seq/2/ >> ./Result/Yelp/Att2Seq/Yelp_Att2Seq_2.log

python -u Att2Seq/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/3/ \
--checkpoint ./Result/Yelp/Att2Seq/3/ >> ./Result/Yelp/Att2Seq/Yelp_Att2Seq_3.log

python -u Att2Seq/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/4/ \
--checkpoint ./Result/Yelp/Att2Seq/4/ >> ./Result/Yelp/Att2Seq/Yelp_Att2Seq_4.log

python -u Att2Seq/main.py --cuda \
--data_path ./Data/Yelp/reviews.pickle \
--index_dir ./Data/Yelp/5/ \
--checkpoint ./Result/Yelp/Att2Seq/5/ >> ./Result/Yelp/Att2Seq/Yelp_Att2Seq_5.log

###################### Amazon - ClothingShoesAndJewelry  #############################
# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/1/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/1/ >> ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/Amz_CSJ_Att2Seq_1.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/2/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/2/ >> ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/Amz_CSJ_Att2Seq_2.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/3/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/3/ >> ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/Amz_CSJ_Att2Seq_3.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/4/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/4/ >> ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/Amz_CSJ_Att2Seq_4.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
# --index_dir ./Data/Amazon/ClothingShoesAndJewelry/5/ \
# --checkpoint ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/5/ >> ./Result/Amazon/ClothingShoesAndJewelry/Att2Seq/Amz_CSJ_Att2Seq_5.log

###################### Amazon - MoviesAndTV  #############################
# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV/1/ \
# --checkpoint ./Result/Amazon/MoviesAndTV/Att2Seq/1/ >> ./Result/Amazon/MoviesAndTV/Att2Seq/Amz_MT_Att2Seq_1.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV/2/ \
# --checkpoint ./Result/Amazon/MoviesAndTV/Att2Seq/2/ >> ./Result/Amazon/MoviesAndTV/Att2Seq/Amz_MT_Att2Seq_2.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV/3/ \
# --checkpoint ./Result/Amazon/MoviesAndTV/Att2Seq/3/ >> ./Result/Amazon/MoviesAndTV/Att2Seq/Amz_MT_Att2Seq_3.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV/4/ \
# --checkpoint ./Result/Amazon/MoviesAndTV/Att2Seq/4/ >> ./Result/Amazon/MoviesAndTV/Att2Seq/Amz_MT_Att2Seq_4.log

# python -u Att2Seq/main.py --cuda \
# --data_path ./Data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir ./Data/Amazon/MoviesAndTV/5/ \
# --checkpoint ./Result/Amazon/MoviesAndTV/Att2Seq/5/ >> ./Result/Amazon/MoviesAndTV/Att2Seq/Amz_MT_Att2Seq_5.log