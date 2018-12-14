---
layout: post
title: Bayesian network
---

### Bayesian Network (Graphical Model)
-----
Usually, most of machine learning models can be represented as a conditional probability as like :
 > P( **"Something I want to know" \|  "Our data"** )

So, Bayesian Network is  just a probabilistic graphical model which can represent some conditional probabilities. We already know many problem can be considered as conditional probabilities. Upside notation is conditional probability of **"Something I want to know"** when **"Our data"** given as well. In statistics,  conditional probability everywhere. The example of bayesian network is below.

{% mermaid %}
graph TD
A((X1)) --> B((Y));
C((X2)) --> B;
D((X3)) --> B;
{% endmermaid %}

This graph indicates conditional probability : $p(Y|X_1,X_2,X_3)$  So, each node of the graph is random variables and directional edges are conditional dependencies.

But, why we need to make these graphical structure?


### Reducing computation with proper conditional probability structure
-----
Let assume our problem is that predicting mood of girl friend before we will meet her. We can consider 6 parameters. 

- 1 Weather of today(**Weather**) = { Rainy, Sunny, Cloudy } 
- 2 My Arthritis(**Knee**) = { Painful, Stable }
- 3 Today`s date course(**Course**) = { Nice , Suck }
- 4 Dining reservation cost(**Cost**) = { \$, \$\$, \$\$\$ }
- 5 Her mood at 1 week ago(**Past mood**) = { GOOD, BAD }
- 6 Whether or IU today released an album(**IU**) = { YES, NO }

- **Our Final Goal(Mood)** : "Her Daily Mood" = { GOOD, BAD }
 
You might think why your arthritis statement is a consideration for this problem, but it will be highly related to weather of today since I have felt pain on my knee almost every rainy time. Of course, our emotional girlfriend also will feel bad if the weather will not OK. And we already know our adorable girl friend always forget her mood after 1 week. Yes, what an adorable women. And sixth parameter, about South Korean singer IU, might be no relation with mood of my girlfriend even if IU is super adorable.

Anyway, to predict her daily emotion, here is your equation.

$$
\sum_{All PossibleParameters}{P(Mood = good, Weather, Knee, Course, Cost, PastMood, IU)}
$$

And the graphical structure of upper equation is:
{% mermaid %}
graph TD
A((Weather)) --> Y((MOOD));
B((Knee)) --> Y;
C((Course)) --> Y;
D((Cost)) --> Y;
E((PastMood)) --> Y;
F((IU)) --> Y;
{% endmermaid %}

Well, there are so many possible cases for this equation actually. 3 possible weather * 2 possible my knee statements, ... , and so on. So, the number of all possible cases is (3 * 2 * 2 * 3 * 2 * 2) = 144. I do not want to compute this complex information just to predict her daily mood.

However, we already know that some parameters are highly related to each other. My horrible pain on knee depend on weather somehow, quality of date course might depend on dining reservation cost. And of course, I just realize our result does not give a shit about  IU`s album release. Let adjust our graph with our new knowledge.

{% mermaid %}
graph TD
A((Weather)) --> Y;
B((Knee)) --> A;
C((Course)) --> Y;
D((Cost)) --> C;
E((PastMood)) --> Y;
F((IU))
Y((MOOD))
{% endmermaid %}

We will check this adjusted graph is helpful for our computation or not. How can we change our prior equation with this new-graph structure? We can rewrite our original equation by using chain rule. $P(X,Y,Z)=P(X|Y,Z)P(Y,Z)=P(X|Y,Z)P(Y|Z)P(Z)$. And of course if there are proper conditional probabilities, we can remove some random variable for each probability. We call it conditional independence. If when Z is given then X and Y independence, upper equation will change : $P(X|Z)P(Y|Z)P(Z)$. Y has gone. This elimination is a main reason to use conditional independence.

$$
P(Mood, Weather, Knee, Course, Cost, PastMood , IU)\\
=P(Mood|Weather,Knee,Course,Cost,PastMood,IU)\\P(Weather,Knee,Course,Cost,PastMood,IU)\\
=P(Mood|Weather,Knee,Course,PastMood)\\P(Weather,Knee,Course,Cost,PastMood,IU)\\
=P(Mood|Weather,Course,PastMood)\\P(Weather|Knee,Course,Cost,PastMood,IU)\\P(Knee,Course,Cost,PastMood,IU)\\
=P(Mood|Weather,Course,PastMood)P(Weather|Knee)\\P(Knee|Course,Cost,PastMood,IU)P(Course,Cost,PastMood,IU)\\
=P(Mood|Weather,Course,PastMood)P(Weather|Knee)\\P(Knee)P(Course|Cost,PastMood,IU)P(Cost,PastMood,IU)\\
=P(Mood|Weather,Course,PastMood)P(Weather|Knee)\\P(Knee)P(Course|Cost)P(Cost,PastMood,IU)\\
$$
$$
=P(Mood|Weather,Course,PastMood)P(Weather|Knee)\\P(Knee)P(Course|Cost)P(Cost)P(PastMood)P(IU)\\
$$

The number of computation 3 * 2 * 2 * 3 * 2 * 2 = 144 is changed to (3 * 2 * 2) + ( 2 ) + ( 3 ) = 17. From now on, I do not have to be slapped by my girlfriend because of her bad mood. I just predict a probability and I will cancel the date if her mood would be bad.
