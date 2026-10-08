# 13-Fact Forced Choice

| Model | Top-1 | Mean correct margin |
|---|---|---|
| Compute5 | 4/13 | -0.5349 |
| Narrative | 3/13 | -0.5611 |
| Contrastive | 4/13 | -0.2288 |
| 50M | 7/13 | 0.0056 |

| Model | Fact | Rank | Margin | Top-1 |
|---|---|---|---|---|
| Compute5 | name_object_location_goal:character_name | 1 | 0.3236 | True |
| Compute5 | name_object_location_goal:object_possession | 1 | 0.1343 | True |
| Compute5 | name_object_location_goal:location | 1 | 1.6162 | True |
| Compute5 | name_object_location_goal:stated_goal | 2 | -2.5403 | False |
| Compute5 | relationship_secret:character_name | 3 | -1.3322 | False |
| Compute5 | relationship_secret:relationship | 1 | 0.1255 | True |
| Compute5 | relationship_secret:secret | 2 | -0.6698 | False |
| Compute5 | relationship_secret:location | 3 | -1.9866 | False |
| Compute5 | possession_scene_detail_goal:character_name | 3 | -1.3181 | False |
| Compute5 | possession_scene_detail_goal:object_possession | 2 | -0.0195 | False |
| Compute5 | possession_scene_detail_goal:location | 2 | -0.9601 | False |
| Compute5 | possession_scene_detail_goal:scene_detail | 2 | -0.1425 | False |
| Compute5 | possession_scene_detail_goal:stated_goal | 2 | -0.1848 | False |
| Narrative | name_object_location_goal:character_name | 2 | -0.3268 | False |
| Narrative | name_object_location_goal:object_possession | 1 | 0.2630 | True |
| Narrative | name_object_location_goal:location | 1 | 1.7179 | True |
| Narrative | name_object_location_goal:stated_goal | 2 | -1.7754 | False |
| Narrative | relationship_secret:character_name | 2 | -1.3999 | False |
| Narrative | relationship_secret:relationship | 1 | 0.0517 | True |
| Narrative | relationship_secret:secret | 2 | -0.5849 | False |
| Narrative | relationship_secret:location | 2 | -1.8364 | False |
| Narrative | possession_scene_detail_goal:character_name | 3 | -0.9550 | False |
| Narrative | possession_scene_detail_goal:object_possession | 3 | -0.4331 | False |
| Narrative | possession_scene_detail_goal:location | 3 | -0.6903 | False |
| Narrative | possession_scene_detail_goal:scene_detail | 3 | -0.5579 | False |
| Narrative | possession_scene_detail_goal:stated_goal | 2 | -0.7678 | False |
| Contrastive | name_object_location_goal:character_name | 2 | -0.0436 | False |
| Contrastive | name_object_location_goal:object_possession | 1 | 0.7006 | True |
| Contrastive | name_object_location_goal:location | 1 | 0.7225 | True |
| Contrastive | name_object_location_goal:stated_goal | 2 | -1.4633 | False |
| Contrastive | relationship_secret:character_name | 2 | -0.3826 | False |
| Contrastive | relationship_secret:relationship | 1 | 0.0664 | True |
| Contrastive | relationship_secret:secret | 2 | -0.5366 | False |
| Contrastive | relationship_secret:location | 2 | -0.5329 | False |
| Contrastive | possession_scene_detail_goal:character_name | 1 | 0.6213 | True |
| Contrastive | possession_scene_detail_goal:object_possession | 2 | -0.5133 | False |
| Contrastive | possession_scene_detail_goal:location | 2 | -0.4556 | False |
| Contrastive | possession_scene_detail_goal:scene_detail | 3 | -0.5679 | False |
| Contrastive | possession_scene_detail_goal:stated_goal | 2 | -0.5900 | False |
| 50M | name_object_location_goal:character_name | 1 | 0.7680 | True |
| 50M | name_object_location_goal:object_possession | 1 | 0.4947 | True |
| 50M | name_object_location_goal:location | 1 | 2.4883 | True |
| 50M | name_object_location_goal:stated_goal | 2 | -1.9121 | False |
| 50M | relationship_secret:character_name | 2 | -0.1838 | False |
| 50M | relationship_secret:relationship | 1 | 0.1190 | True |
| 50M | relationship_secret:secret | 2 | -0.3433 | False |
| 50M | relationship_secret:location | 2 | -1.5871 | False |
| 50M | possession_scene_detail_goal:character_name | 1 | 0.9955 | True |
| 50M | possession_scene_detail_goal:object_possession | 1 | 0.3879 | True |
| 50M | possession_scene_detail_goal:location | 2 | -0.7701 | False |
| 50M | possession_scene_detail_goal:scene_detail | 2 | -0.5719 | False |
| 50M | possession_scene_detail_goal:stated_goal | 1 | 0.1880 | True |
