# Questions:

1. What other approaches could you have chosen to tackle this task and why did you settle on the approach you chose?

Approaches I'd consider, given the playing style isn't something that's labelled:
    - Prescribing scenarios manually that are widely recognised, and using a rules based approach. This seems to be what StatsPerform has done. "Football First".
    - Unsupervised learning that uses all features, that is then narrowed down to features of variance. (I don't like this idea).
    - There's a great talk about Graph Convolutional Networks that was at Statsbomb conference, using 360 data to build a graph to incorporate location context, then adding features to that and surrounding events, trained on predicting the event, which seemed good for a labelling tool.

    My approach is a mix of football first, in that we're choosing what features we're interested in, and unsupervised clustering, so as not to be too prescriptive: 
    - Unsupervised learning, using chosen and calculated features that describe playing style.
        - Different clustering algorithms, display using things like UMAP, TSNE or PCA.
        - I think the clustering algorithm will be either density based (DBSCAN/OPTICS) or Agglomerative.

2. What are the flaws and limitations of your approach?
    - Have to define a load of features that took a lot of time.
    - Have to find the right combination of features that define clusters of playing styles.
    - Playing styles vary within match, and currently the features I aggregate are for the entire match (although this should change).
    - Can't be used within match (unless a model with sequences is created).
    - Unsupervised techniques are hard to validate.
    - I don't know much about the WSL, limiting my ability to confirm results, and also affects which features I think are important.

3. What insights or metrics could you derive based on the outputs of your model?
    - Effectiveness of playing style, e.g. how does the playing style relate to xGD.
    - Can also use aggregate features for insights, e.g. pass height vs success.
    - Could examine how playing style (defined by features) varies with manager changes.

4. Given infinite time, how might you improve on your approach?
    - Finish this off as a python package, there's plenty to do.
    - Pay an expert/group of experts to label some data, to enable better validation, and enable use of other modelling techniques (supervised classfication).

5. What changes would you make to your implementation before you considered it production-ready?
    - It needs a config file functionality.
    - Improved functions, such that it could be imported and used in a jupyter notebook by someone taking an interest.
    - A defined model at the end would be useful, not in a notebook.
