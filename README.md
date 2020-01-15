## Project4 alpha research multi factor modeling
1.准备数据

使用yfinance包可以直接下载SPY 500的历史数据，需要导出成Sqlite库的.db文件, 供后序Zipline创建自定义Data Bundle使用。

我是通过先下载到mysql数据库，然后转成Sqlite库.db文件，当然可以直接下载insert到Sqlite库中，下载的代码参考 **download-data/main.py**。

2.编写自定义bundle

进入**project4-alpha_research_multi_factor_modeling/** 下，创建extension.py编辑对应逻辑, 同时将准备好的Sqlite库.db文件也放置到同一目录下。

然后设置ZIPLINE_ROOT环境变量为当前目录，同时还需要在该目录中创建data目录，后面ingest生成的文件都会生成在data文件夹中。

3.运行项目

## Project5 NLP on financial statements
1.nltk

国内下载nltk太慢的话，可以手动下载nltk包，详细参考：http://www.nltk.org/data.html ，文末`Manual installation`模块详述了如何手动安装到指定目录

2.pricing data

pricing data使用的是上一个项目的db文件

3.项目在原来的基础上做了一些优化，包括但不限于：优化内存；10-k文件下载后本地化等，欢迎提供更多的优化建议和有关项目上的建议、意见和知识分享

<b>Contact: rongjing_xue@163.com
</b>
