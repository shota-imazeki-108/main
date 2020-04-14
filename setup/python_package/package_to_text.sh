today=`date "+%Y%m%d"`
mkdir -p ./list/pip ./list/conda
pip freeze --all -> ./list/pip/${today}'.txt'
conda list --export > ./list/conda/${today}'.txt'