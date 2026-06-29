sudo apt install unzip
curl -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d ./datasets
rm DIV2K_train_HR.zip
mv ./datasets/DIV2K_train_HR ./datasets/DIV2K_HR
