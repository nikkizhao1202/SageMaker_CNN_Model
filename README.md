# Training Tweets Sentiment Analysis CNN Model Running on SageMaker NoteBook

<ul>
  <li> Contributor: Yongyi(Nikki)Zhao</li>
</ul>

## Overview
<p> 
  CNN Tweets Sentiment Model Training Done on AWS SageMaker
</p >

## Details

<ul>
  <li> Create SageMaker Notebook Instance</li>
  <li> Create S3 Bucket and Upload File and Data to id </li>
  <li> Modify Directory to be able to load file from your own S3 Bucket</li> 
  <p> <strong> Enviorment you need to change: </strong> <br/>
    1. train directory <br/>
    2. validation directory <br/>
    3. test directory <br/>
    4. dictionary directory<br/>
    <strong> My enviroenment is as following: </strong>
 
   ```sh
       Train_Dir = 'S3://twitter-text/training/train/'
   ```
   ```sh
       Val_Dir = 'S3://twitter-text/training/dev/'
   ```
   ```sh
       Test_Dir = 'S3://twitter-text/training/eval/'
   ```
   ```sh
       Dic_Dir = 'S3://twitter-text/training/data/'
  ```
  </p>
  
<span style="color:blue"> Due to git capacity issue, we didn't upload glove dictionary, but you can click the link in Reference Matrial Section to download the dictionary and run locally </span> 

## Requirements
<ul>
  <li> Python (3.6, 3.7) </li>
  <li> TensorFlow v1.13 </li>
  <li> AWS Account </li>
  
</ul>

## Reference Matrial
- [Twitter GloVe Dict](https://twitter-text.s3.amazonaws.com/training/data/glove.twitter.27B.25d.txt) 



## Discussion and Development

<p> Most development discussion is taking place on github in this repo.</p >

## Contributing to Tweets Preprocessing Library
<p>
Any contributions, bug reports, bug fixes, documentation improvements, enhancements to make this project better are warmly welcomed.
</p >
