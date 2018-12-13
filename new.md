**Machine learning problem with graphical model**
---


### Bayesian Network (Graphical Model)
-----
Usually, most of machine learning models can be represented as a conditional probability as like :
 > P( **"Something I want to know" |  "Our data"** )

So, Bayesian Network is  just a probabilistic graphical model which can represent some conditional probabilities. We already know many problem can be considered as conditional probabilities. Upside notation is conditional probability of **"Something I want to know"** when **"Our data"** given as well. In statistics,  conditional probability everywhere.



But, why we need to make these graphical structure?


### Reducing computation with proper conditional probability structure
-----
Let assume our problem is that predicting mood of girl friend before we will meet her. We can consider 5 parameters. 

-  Weather of today(**Weather**) = { Rainy, Sunny, Cloudy } 
-  My Arthritis(**Knee**) = { Painful, Stable }
-  Today`s date course(**Course**) = { Nice , Suck }
-  Dining reservation cost(**Cost**) = { \$, \$\$, \$\$\$ }
-  Her mood at 1 week ago(**Past mood**) = { GOOD, BAD }

- **Our Final Goal(Mood)** : "Her Daily Mood" = { GOOD, BAD }
 
You might think why your arthritis statement is a consideration for this problem, but it will be highly related to weather of today since I have felt pain on my knee almost every rainy time. Of course, our emotional girlfriend also will feel bad if the weather will not OK. And we already know our adorable girl friend always forget her mood after 1 week. Yes, what an adorable women.

Anyway, to predict her daily emotion, here is your equation.

Well, there are so many possible cases for this equation actually. 3 possible weather * 2 possible my knee statements, ... , and so on. So, the number of all possible cases is (3*2*2*3*2) = 72. I do not want to compute this complex informations just to predict her daily mood.

However, we already know that some parameters are highly related to each other. My horrible pain on knee depend on weather somehow, quality of date course might depend on dining reservation cost.
