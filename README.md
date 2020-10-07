#      

#                       灾难应对管道项目

#        （Disaster Response Pipeline Project）

**By Yaping**



​        在本项目中，我们将应用数据工程技术分析来自 [**Figure 8**](https://www.figure-eight.com/) 的灾害消息，构建一个分类灾害消息的模型并用于 API。我们根据包含灾害发生时发出的真实消息的数据集，创建一个机器学习管道对这些事件进行分类，使得最终可以将消息发送到合适的灾害应对机构。本项目还包括一个网络应用程序，灾害应对工作人员能够输入一条新消息然后得到分类结果，网络应用程序还会对数据进行可视化展示。

###        项目主要包括三部分：

**  1.ETL** **管道**

​       程序在 Python 脚本 process_data.py 中，是一个数据清洗管道。

**  2.****机器学习管道**

​       程序在 Python 脚本 train_classifier.py 中，是一个机器学习管道。

**  3.Flask** **网络应用程序**

​      这是一个 Flask 网络应用程序，可以使用 Flask 、HTML、CSS 和 JavaScript 知识自由地添加新特征。在本部分，主要是实现项目可视化功能。



###          可视化截图：

![屏幕快照 2020-10-07 上午10.37.49](/Users/zhu/Desktop/屏幕快照 2020-10-07 上午10.37.49.png)





![屏幕快照 2020-10-07 上午10.40.43](/Users/zhu/Desktop/屏幕快照 2020-10-07 上午10.40.43.png)

![屏幕快照 2020-10-07 上午10.41.56](/Users/zhu/Desktop/屏幕快照 2020-10-07 上午10.41.56.png) 



### 程序运行指示:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
