## set up
---
```
pip install doc2dash
pip install breathe
pip install sphinx_rtd_theme
brew install doxygen
```
## docset add
---
```
git clone hoge
cd hoge/docs
make html
doc2dash -n LightGBM -a -d ~/Library/Application\ Support/Dash/DocSets/LightGBM _build/html
# アイコン設定する場合は -iを入れる
```

## appendix
---
```
https://kapeli.com/dash
```