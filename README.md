#### Eric Bai
#### 12/20/2015

# Searching by Category: a search engine with user input category for wikipedia

## Motivation

There are often times when people search a key word in wikipedia, the return page is a list of
possible meanings/links for user to choose from. However, if a user use combination of both the key word and the category, wikipedia are not able to return the related results immediately. Instead, it will still return a page of lists of possible links. For example, search results for [recall](https://en.wikipedia.org/wiki/Recall) and [recall in statistcs](https://en.wikipedia.org/w/index.php?search=recall+in+statistics&title=Special%3ASearch&go=Go) in wikipedia is as following.The goal is to load [recall in statistics](https://en.wikipedia.org/wiki/Precision_and_recall) directly.
![](img/recall_example.jpg)
![](img/recall_in_statistics_example.jpg)

## Data Pipeline and Modeling Steps


* Start with small wikipedia data on S3 (s3n://wikisample10/sample2)
* Use TF-IDF and matrix factorization for topic modeling to group articles into categories and then use Naive Bayes to predict a given category in which group
* Calculate cosine similarities with user input category for all pages in the same keyword group page
* Based on the closet category search and the highest cosine similarity to rank relevant pages
* Run the model on the whole dataset
* Build a usable web app


#### Potential probalems:

* It is hard to pickle models in mllib


### References

* Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." the Journal of machine Learning research 3 (2003): 993-1022.

* Page, Lawrence, et al. "The PageRank citation ranking: bringing order to the Web." (1999).
