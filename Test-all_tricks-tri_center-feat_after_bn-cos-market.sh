# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# without re-ranking
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/home/itsnamgyu/data')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home/itsnamgyu/code/reid-strong-baseline/outputs/market_resnet50_model_120.pth')"