## MVP
The safety of the pilgrims is one of the most important things in the Kingdom of Saudi Arabia and in the Islamic world in general, so we will create a system capable of managing crowds by determining number of people in the holy places in order to avoid overcrowding accidents.

First, we thought about the data, since it is not available, so we decided to take data include pictures of crowds from around the world in order to measure them
Second, we took some time to search for the best model that we can use to measure traffic, and we found that CSRnet is the best model for traffic measurement because it Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes. 
We initially applied it to crowded images in general and got excellent results such as  

![Picture1](https://user-images.githubusercontent.com/71217830/148990859-cfb0750e-aee1-4f38-9029-dd1d3022b931.jpg)
![Picture2](https://user-images.githubusercontent.com/71217830/148990927-84ac11d3-00c6-4135-9034-e3567a4696b8.png)

Predict : 362
True : 361

We hope that we will get excellent results when applying it to pictures of the holy places in Makkah so that we can organize the crowds there
