# Questions:



1. What other approaches could you have chosen to tackle this task and why did you settle on the approach you chose?

Approaches I'd consider, given the playing style isn't something that's labelled:
    - Prescribing scenarios manually that are widely recognised, and using a rules based approach. This seems to be what StatsPerform has done.
    - Unsupervised learning that uses all features, that is then narrowed down to features of variance. (I don't like this idea).

    - Unsupervised learning, using chosen features that describe playing style.
        - Different clustering algorithms, display using things like UMAP, TSNE or PCA.
        - I think the clustering algorithm will be either density based (DBSCAN/OPTICS) or Agglomerative.

2. What are the flaws and limitations of your approach?
    - Playing styles are dynamic and teams with similar playing styles may implement them differently, or change playing style within a game, within a season, and against different oppositions. 
    - Unsupervised techniques are hard to validate.
    - Using lots of features is tempting but gets harder to separate teams

3. What insights or metrics could you derive based on the outputs of your model?
    - Effectiveness of playing style, e.g. how does the playing style relate to xGD and therefore game management optimisation.
    - Manager impact

4. Given infinite time, how might you improve on your approach?
    - Pay an expert/group of experts to label the data, to enable better validation, and enable use of other modelling techniques (supervised classfication).

5. What changes would you make to your implementation before you considered it production-ready?
    - Hoping to get things out of notebooks quickly
    - Get it on the cloud