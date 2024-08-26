# Training results

## Gender, 2 classes 

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.0649652845712542,                                  
"train_loss_std": 0.007431198822804523,                                 
"train_accuracy_mean": 0.9799522909337441,                              
"train_accuracy_std": 0.002990603883655691,                             
"val_loss_mean": 0.1180148920946893,                                    
"val_loss_std": 0.015007247824669136,                                   
"val_accuracy_mean": 0.9638793030305295,                                
"val_accuracy_std": 0.006335605671980757,                               
"test_loss_mean": 0.43533751480107485,                                  
"test_loss_std": 0.0778508579963934,                                    
"test_accuracy_mean": 0.8406407172280743,                               
"test_accuracy_std": 0.027314883353706715
```

- time elapsed: 13 mins

#### cross-val on adience, pretrained on imdb_wiki & AgeDB & UTKFace

```
"train_loss_mean": 0.033158706331651626,                                
"train_loss_std": 0.0020978169676838273,                                
"train_accuracy_mean": 0.9916501015955475,                              
"train_accuracy_std": 0.0008641596949750719,                            
"val_loss_mean": 0.08097766418782043,                                   
"val_loss_std": 0.01669467976343782,                                    
"val_accuracy_mean": 0.9771785595684266,                                
"val_accuracy_std": 0.004885199751184193,                               
"test_loss_mean": 0.31197317347965897,                                  
"test_loss_std": 0.0603604492284769,                                    
"test_accuracy_mean": 0.8887175397758311,                               
"test_accuracy_std": 0.025486444846123855  
```

- time elapsed: 19 mins

### ResNet model

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.04592202911063537,                                 
"train_loss_std": 0.004968584913553469,                                 
"train_accuracy_mean": 0.9874470878705587,                              
"train_accuracy_std": 0.0020352231866869945,                            
"val_loss_mean": 0.10436273033332649,                                   
"val_loss_std": 0.010926707221016262,                                   
"val_accuracy_mean": 0.9672422859006637,                                
"val_accuracy_std": 0.0039099967107252545,                              
"test_loss_mean": 0.4047493383832665,                                   
"test_loss_std": 0.08096214807019218,                                   
"test_accuracy_mean": 0.8412609429948891,                               
"test_accuracy_std": 0.03257054909731762 
```

- time elapsed: 11 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.04001438188234187,                                 
"train_loss_std": 0.002662775959211709,                                 
"train_accuracy_mean": 0.9898581723087919,                              
"train_accuracy_std": 0.0013010347725655978,                            
"val_loss_mean": 0.08413476834629004,                                   
"val_loss_std": 0.01292393044734325,                                    
"val_accuracy_mean": 0.975305017750686,                                 
"val_accuracy_std": 0.004178405528355561,                               
"test_loss_mean": 0.2767549877755469,                                   
"test_loss_std": 0.04573368326324859,                                   
"test_accuracy_mean": 0.8985643474512477,                               
"test_accuracy_std": 0.019786984421465842 
```

- time elapsed: 11 mins

### VGG model

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.18788806400555194,                                 
"train_loss_std": 0.06184505833485614,                                  
"train_accuracy_mean": 0.9425203519974567,                              
"train_accuracy_std": 0.010204944858290453,                             
"val_loss_mean": 0.20671770950903845,                                   
"val_loss_std": 0.062320597132081997,                                   
"val_accuracy_mean": 0.9335705452192901,                                
"val_accuracy_std": 0.012128948550586624,                               
"test_loss_mean": 0.46219740461201525,                                  
"test_loss_std": 0.09135287489074899,                                   
"test_accuracy_mean": 0.8206668461496278,                               
"test_accuracy_std": 0.03890423963959933
```

- time elapsed: 12 mins

#### cross-val on adience, pretrained on imdb_wiki & AgeDB & UTKFace

```
"train_loss_mean": 0.04375674086876357,                                 
"train_loss_std": 0.0036730700854976346,                                
"train_accuracy_mean": 0.9880428284821821,                              
"train_accuracy_std": 0.001052018075001737,                             
"val_loss_mean": 0.08951315115628447,                                   
"val_loss_std": 0.014845133446288579,                                   
"val_accuracy_mean": 0.9742719259125172,                                
"val_accuracy_std": 0.004708878367366926,                               
"test_loss_mean": 0.2679020066129207,                                   
"test_loss_std": 0.06755369249286816,                                   
"test_accuracy_mean": 0.9066379871650122,                               
"test_accuracy_std": 0.024821142870521833 
```

- time elapsed: 11 mins


## Age, Regression 

### ResNet model

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.018803563352668903,                                
"train_loss_std": 0.003981359137520785,                                 
"train_mae_mean": 0.9968163529584783,                              
"train_mae_std": 0.0007963574511976345,                            
"val_loss_mean": 0.23540298709051374,                                   
"val_loss_std": 0.02706932878500526,                                    
"val_mae_mean": 0.9343246823460251,                                
"val_mae_std": 0.007345127657187889,                               
"test_loss_mean": 1.6534216372526924,                                   
"test_loss_std": 0.17353274901325103,                                   
"test_mae_mean": 5.7,                                
"test_mae_std": 0.03969608603733163
```

- time elapsed: 13 mins

#### cross-val on adience, pretrained on imdb_wiki & AgeDB & UTKFace

```
"train_loss_mean": 0.04545511317544952,                                 
"train_loss_std": 0.011582151765869825,                                 
"train_mae_mean": 0.9912913328381187,                              
"train_mae_std": 0.0027117227668752996,                            
"val_loss_mean": 0.26228065723772653,                                   
"val_loss_std": 0.027238944790212685,                                   
"val_mae_mean": 0.919461702405183,                                 
"val_mae_std": 0.008025233795177144,                               
"test_loss_mean": 1.3636066654437713,                                   
"test_loss_std": 0.06940663370589772,                                   
"test_mae_mean": 5.7,                                
"test_mae_std": 0.023349385815310295 
```

- time elapsed: 14 mins


#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.017032788165818446,                                
"train_loss_std": 0.004706186021496336,                                 
"train_mae_mean": 0.9972558031465577,                              
"train_mae_std": 0.0008681205524060541,                            
"val_loss_mean": 0.23448244291745723,                                   
"val_loss_std": 0.027292528189579012,                                   
"val_mae_mean": 0.9325784941969336,                                
"val_mae_std": 0.007301875082802385,                               
"test_loss_mean": 1.6318892459315955,                                   
"test_loss_std": 0.15858648723166602,                                   
"test_mae_mean": 5.7,                                
"test_mae_std": 0.035446737871884244  
```

- time elapsed: 11 mins

#### cross-val on adience, pretrained on imdb_wiki & AgeDB & UTKFace

```
"train_loss_mean": 0.04437869559662234,                                 
"train_loss_std": 0.012701954716739005,                                 
"train_mae_mean": 0.9918182653022005,                              
"train_mae_std": 0.0028055626912445683,                            
"val_loss_mean": 0.26793492241125505,                                   
"val_loss_std": 0.027698279176450572,                                   
"val_mae_mean": 0.919607849371705,                                 
"val_mae_std": 0.007822749846796135,                               
"test_loss_mean": 1.4226110422261125,                                   
"test_loss_std": 0.14027027999068215,                                   
"test_mae_mean": 5.7,                                
"test_mae_std": 0.03617848265492994 
```

- time elapsed: 11 mins

### VGG model

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.453935819426141,                                   
"train_loss_std": 0.047157416142341924,                                 
"train_mae_mean": 0.8339816696791367,                              
"train_mae_std": 0.019246473110951536,                             
"val_loss_mean": 0.5811139679108489,                                    
"val_loss_std": 0.04081904610737358,                                    
"val_mae_mean": 0.7932559053275473,                                
"val_mae_std": 0.018348902420289123,                               
"test_loss_mean": 1.3348493398058838,                                   
"test_loss_std": 0.14247212927363231,                                   
"test_mae_mean": 5.7,                                
"test_mae_std": 0.03716699659878682 
```

- time elapsed: 18 mins

#### cross-val on adience, pretrained on imdb_wiki & AgeDB & UTKFace

```
"train_loss_mean": 0.10999468074281796,                                 
"train_loss_std": 0.021556805519474653,                                 
"train_mae_mean": 0.9726441508684591,                              
"train_mae_std": 0.006406822795072672,                             
"val_loss_mean": 0.34819507628122076,                                   
"val_loss_std": 0.02380311932283215,                                    
"val_mae_mean": 0.8930195217194301,                                
"val_mae_std": 0.0065059300009134,                                 
"test_loss_mean": 1.3879593566341297,                                   
"test_loss_std": 0.07896728725735712,                                   
"test_mae_mean": 5.7,                                
"test_mae_std": 0.021638053524286733 
```

- time elapsed: 10 mins


#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.044960734909356724,                                
"train_loss_std": 0.007755128736973053,                                 
"train_mae_mean": 0.990803508707438,                              
"train_mae_std": 0.0020723646887524454,                            
"val_loss_mean": 0.26500815759495355,                                   
"val_loss_std": 0.02852086799293112,                                    
"val_mae_mean": 0.9188075365091344,                                
"val_mae_std": 0.008018510317906943,                               
"test_loss_mean": 1.4405305576859275,                                   
"test_loss_std": 0.08819631245400522,                                   
"test_mae_mean": 5.7,                                
"test_mae_std": 0.02257798885132503   
```

- time elapsed: 15 mins

#### cross-val on adience, pretrained on imdb_wiki & AgeDB & UTKFace 

```
"train_loss_mean": 0.01984713265981791,                                 
"train_loss_std": 0.0049036925402442125,                                
"train_accuracy_mean": 0.9965252429269149,                              
"train_accuracy_std": 0.0008567829984185799,                            
"val_loss_mean": 0.2336303363418904,                                    
"val_loss_std": 0.027047535744339275,                                   
"val_accuracy_mean": 0.9326447526841612,                                
"val_accuracy_std": 0.008075767818661417,                               
"test_loss_mean": 1.6528112440090832,                                   
"test_loss_std": 0.16714354925780017,                                   
"test_accuracy_mean": 0.5434543605071928,                               
"test_accuracy_std": 0.037108616301812565  
```

- time elapsed: 13 mins
