#Alex Eliseev
#This Python script runs binary classification of fake/real news. The main function is called upon by the main
#jupyter notebook file for Q-learning and also for general model training/evaluation. It is told how many artificially generated
#fake news articles it should use to augment the fake news portion of the training dataset. The actual fake news and real news it simply pulls
#from the training dataset folders. It trains then evaluates the classifier on the test set of fake/real news that was set aside.
#The classifier is a BERT transformer model from one of the labs, it is quite good for this task. Labeling is done as the articles are loaded from their folders.
#Lastly it has an option to use all or a smaller training set. Generating fake articles took a lot of time and we did not make that many.
#To make their impact more apperent a smaller dataset could be used with a custom number of real fakes/real articles.
def TrainAndEvaluateClassifer(NumberOfArtificialArticlesTouse,UseAll,NumberOfRealFakeArticles=400,NumberOfRealArticles=800):
    #imports
    import os
    import random

    #Array will hold all of the labeled fake, generated fake and real articles.
    AllArticles = [] 

    #First loading all of the real articles.
    RealArticlesFilePath = "DS8008FinalProject/Data/train/RealNewsArticlesTrainingSet/"
    RealArticles = os.listdir(RealArticlesFilePath)
    random.shuffle(RealArticles)

    if(UseAll == 0): #If this option enabled, using only limited number of real articles. 
        RealArticles = RealArticles[:NumberOfRealArticles]
    
    for Article in RealArticles:
        FilePath = os.path.join(RealArticlesFilePath,Article)
        
        #Opening article from the training folder real news articles, cleaning it slightly and saving it to array with label 0 for real
        with open(FilePath,"r",encoding="utf-8",errors="ignore") as File:
            Text = File.read()
            Text = Text.replace("<br />"," ").replace("<br>"," ")
            AllArticles.append((Text,0)) #Appending with label 0 in 2nd column for real article.

    #Repeating same process, loading all of the real fake articles from their folder into the training array now
    FakeArticlesFilePath = "DS8008FinalProject/Data/train/FakeNewsArticlesTrainingSet/"
    FakeArticles = os.listdir(FakeArticlesFilePath)
    random.shuffle(FakeArticles)

    if(UseAll == 0): #If this option enabled, using only 400 fake articles. 
        FakeArticles = FakeArticles[:NumberOfRealFakeArticles]
    
    for Article in FakeArticles:
        FilePath = os.path.join(FakeArticlesFilePath,Article)
        
        #Opening article from the training folder real news articles, cleaning it slightly and saving it to array with label 0 for real
        with open(FilePath,"r",encoding="utf-8",errors="ignore") as File:
            Text = File.read()
            Text = Text.replace("<br />"," ").replace("<br>"," ")
            AllArticles.append((Text,1)) #Appending with label 1 for fake article.
    
    #Lastly repeating same process but loading specified amount of artificially generated articles from their folder into the trainign array.
    GeneratedFakeArticlesFilePath = "DS8008FinalProject/Data/train/FakeNewsArticlesArtificiallyGeneratedTrainingSet/"
    GeneratedFakeArticles = os.listdir(GeneratedFakeArticlesFilePath)
    random.shuffle(GeneratedFakeArticles)

    #Using only specified amount or all if amount exceeds available
    GeneratedFakeArticles = GeneratedFakeArticles[:NumberOfArtificialArticlesTouse]
    
    for Article in GeneratedFakeArticles:
        FilePath = os.path.join(GeneratedFakeArticlesFilePath,Article)
        
        #Opening article from the training folder real news articles, cleaning it slightly and saving it to array with label 0 for real
        with open(FilePath,"r",encoding="utf-8",errors="ignore") as File:
            Text = File.read()
            Text = Text.replace("<br />"," ").replace("<br>"," ")
            AllArticles.append((Text,1)) #Appending with label 1 for fake article.

    #AllArticles[] now filled with Column 1 = Article Text Column 2 = Associated Label
     




