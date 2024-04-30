class univariate:
    def read_excel(excel):
        import pandas as pd 
        dataset = pd.read_csv(excel)
        return dataset

    def qualquan(dataset):
        Qual = []
        Quan = []
        for i in dataset.columns:
            if (dataset[i].dtypes == "O"):
                Qual.append(i)
            else:
                Quan.append(i)
        print(Qual)
        print(Quan)
        return Qual,Quan

    def unitable(dataset,Quan):
        import pandas as pd
        import numpy as np
        Descriptive = pd.DataFrame(index = ["Mean", "Median", "Mode","Q1:25(percentile)","Q1:25(describe)","Q2:50","Q3:75","99%","Q4:100","IQR","1.5rule","Lesser","Greater","Min","Max"],columns = Quan)
        for i in Quan:
            Descriptive[i]["Mean"] = dataset[i].mean()
            Descriptive[i]["Median"] = dataset[i].median()
            Descriptive[i]["Mode"] = dataset[i].mode()[0] 
            Descriptive[i]["Q1:25(percentile)"] = np.percentile(dataset[i],25)
            Descriptive[i]["Q1:25(describe)"] = dataset.describe()[i]["25%"]
            Descriptive[i]["Q2:50"] = dataset.describe()[i]["50%"]
            Descriptive[i]["Q3:75"] = dataset.describe()[i]["75%"]
            Descriptive[i]["99%"] = np.percentile(dataset[i],99)
            Descriptive[i]["Q4:100"] = dataset.describe()[i]["max"]
            Descriptive[i]["IQR"] = Descriptive[i]["Q3:75"] - Descriptive[i]["Q1:25(describe)"]
            Descriptive[i]["1.5rule"] = 1.5 * Descriptive[i]["IQR"]
            Descriptive[i]["Lesser"] = Descriptive[i]["Q1:25(describe)"] - Descriptive[i]["1.5rule"]
            Descriptive[i]["Greater"] = Descriptive[i]["Q3:75"] + Descriptive[i]["1.5rule"]
            Descriptive[i]["Min"] = dataset[i].min()
            Descriptive[i]["Max"] = dataset[i].max()
            Descriptive[i]["Skew"] = dataset[i].skew()
            Descriptive[i]["Kurtosis"] = dataset[i].kurtosis()
            Descriptive[i]["Var"] = dataset[i].var()
            Descriptive[i]["Std"] = dataset[i].std()
        return Descriptive        

    def frequency_table(column_name,dataset):
        import pandas as pd
        FreqTable=pd.DataFrame(columns = ["unique_values","Frequency","relative_frequency","cumsum"])
        FreqTable["unique_values"] = dataset[column_name].value_counts().index
        FreqTable["Frequency"] = dataset[column_name].value_counts().values
        FreqTable["relative_frequency"] = FreqTable["Frequency"]/FreqTable.shape[0]
        FreqTable["cumsum"] = FreqTable["relative_frequency"].cumsum()
        return FreqTable

    def get_pdf_graph(dataset,start,end):
        from matplotlib import pyplot
        from scipy.stats import norm
        import seaborn as sns
        ax = sns.distplot(dataset,kde=True,kde_kws={"color": "blue"},color = "yellow")
        pyplot.axvline(start,color="black")
        pyplot.axvline(end,color="black")
        
        sample=dataset
        mean = sample.mean()
        std = sample.std()
        
        dist = norm(mean,std)
        
        values = [value for value in range(start,end)]
        Probability = [dist.pdf(value) for value in values]
        
        Prob = sum(Probability)
        print("The area density of range from {} to {} : {}".format(start,end,Prob) )
        
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf=ECDF(dataset)
        Iu=ecdf(start)
        return Prob,Iu

    def standard_normal_distribution(dataset):
        sample=dataset
        mean = sample.mean()
        std = sample.std()
        values = [value for value in dataset]
        Single_value = [i for i in values]
        Z_score = [(Single_value - mean)/std]
        import seaborn as sns
        sns.distplot( Z_score)

    

    








         
        

   

