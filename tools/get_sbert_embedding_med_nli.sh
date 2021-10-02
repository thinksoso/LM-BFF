MODEL=$1
K=16
conda activate base

python tools/get_sbert_embedding.py --sbert_model $MODEL --task MED_NLI
python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task MED_NLI

for seed in 13 21 87 100
do
    cp data/k-shot/MNLI/$K-42/test_matched_sbert-$MODEL.npy  data/k-shot/MNLI/$K-$seed/
    cp data/k-shot/MNLI/$K-42/test_mismatched_sbert-$MODEL.npy  data/k-shot/MNLI/$K-$seed/
done
