## Objectives
- Scrape betting data from oddsshark and sportsplays
- Merge tables on team name, dates played
- Train data on W/L of public betting % and certain ranges of spread for home underdog teams
- Predict games
- Profit

# Quick Summary
The main thing this little side project accomplished is the creation of a clean public betting % and spread database for past NFL games within the last 3 years. The main.py script can pull and merge betting information from oddsshark and sportsplays. It  should be able to be modified for baseball and basketball as well.

Depending on the parameters you query, you can find some decent history trends for certain percentages and spreads.

For instance, I was able to find that there is a 70% win rate when home teams are between 0 to +5.5 and the public is 50% or more on the away team.

However for predictive purposes, after training the model I was only getting a little over 50% at best. I have a couple of ROC curves in the model_train.ipynb jupyter notebook file to show true positive vs false positive rates for models trained with Random Forest Classifier and logistic regression.

My initial reasoning for only scraping the spread and public % was that I thought they might have a strong relationship in predicting covers, especially since "fading the public" and "home dogs" are a popular concept in the sportsbetting community. In the future I may add more features after scraping team stats from football-reference, and that might add more predictive power. I would seek to make the model hit 60% or better on test predictions before deploying it.
