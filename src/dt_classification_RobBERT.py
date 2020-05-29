# Run this with python ./src/dt_classification_RobBERT.py

from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# EXPERIMENT 2020-05-28 22:40
#
# Add validations for vinden en lopen and add some more sentences in training set
# Not actually executed on Thursday evening; doing a large scale validation experiment instead

# RESULTS 2020-05-28 18:28 (locally saved in RoBERTa-003)
#
# Add some explicit synthetic training data for `vinden` en `lopen`
#
# This added 1 "regression", where one case of "zenden" in the imperative form.
# Looks like "imperative" is more difficult to detect.
#
# INFO:simpletransformers.classification.classification_model: Training of roberta model complete. Saved to outputs/RoBERTa.
# INFO:simpletransformers.classification.classification_model:{'mcc': 0.9759000729485332, 'tp': 40, 'tn': 41, 'fp': 0, 'fn': 1, 'eval_loss': 0.06506455917075403}
# [[ 4.5127707 -5.189252 ]
#  [ 4.422516  -5.141899 ]
#  [ 4.4376254 -5.060404 ]
#  [ 4.1830378 -4.784006 ]
#  [ 4.48708   -5.177963 ]
#  [ 3.5659823 -4.0448856]
#  [ 4.509686  -5.238132 ]
#  [ 3.2408845 -3.762731 ]
#  [ 4.437847  -5.0309286]
#  [ 4.515825  -5.193684 ]
#  [ 4.564027  -5.113303 ]
#  [ 4.39019   -5.027665 ]
#  [ 4.528157  -5.1643867]
#  [ 4.314218  -4.903183 ]
#  [ 4.485026  -5.120996 ]
#  [-3.8476033  4.5130563]
#  [-4.5768366  5.0869884]
#  [-4.5324497  5.054275 ]
#  [-4.4260025  5.140185 ]
#  [-4.5948057  5.082098 ]
#  [-3.6306684  4.150005 ]
#  [-4.4052696  4.963736 ]
#  [ 2.6459968 -3.069624 ] # fn ("zendt" imperative form mistake not detected)
#  [-4.6243696  5.317634 ]
#  [-4.584487   5.075942 ]
#  [-4.586463   5.059649 ]
#  [-4.6279826  5.463476 ]
#  [-4.532429   5.3678446]
#  [-4.448793   5.161991 ]
#  [-4.5559072  5.0067186]
#  [ 4.579213  -5.2127857]
#  [ 4.376686  -4.977299 ]
#  [ 4.5584974 -5.2642884]
#  [-4.59206    5.324033 ]
#  [-4.632761   5.1654453]
#  [-4.776362   5.431049 ]
#  [ 4.4678965 -5.0713797]
#  [-4.6983333  5.4305263]
#  [ 4.273419  -5.012199 ]
#  [-4.7211637  5.374429 ]
#  [ 4.5263247 -5.2804236]
#  [-4.5844965  5.113065 ]
#  [ 4.494684  -5.2651405]
#  [-4.7487335  5.483908 ]
#  [ 4.5410194 -5.2400646]
#  [-4.607552   5.1105533]
#  [ 4.415904  -5.115765 ]
#  [-4.6453424  5.1263723]
#  [ 4.39664   -4.965718 ]
#  [-4.6048393  5.1230874]
#  [ 4.4536414 -5.1998444]
#  [-4.6293106  5.094838 ]
#  [ 4.5860386 -5.282308 ]
#  [ 4.4852953 -5.177291 ]
#  [ 4.5087943 -5.21908  ]
#  [ 4.5944767 -5.331482 ]
#  [ 4.5319943 -5.2086506]
#  [ 4.587343  -5.27311  ]
#  [ 4.4747114 -5.1729345]
#  [ 4.537204  -5.256872 ]
#  [ 4.5376263 -5.288742 ]
#  [ 4.5452933 -5.2173653]
#  [ 4.6053686 -5.2954736]
#  [ 4.530226  -5.2567606]
#  [ 4.5192127 -5.243206 ]
#  [ 4.5766706 -5.2628508]
#  [ 4.629884  -5.191097 ]
#  [-4.714321   5.349707 ]
#  [-4.717781   5.4232655]
#  [-4.6720243  5.3952346]
#  [-4.7530055  5.501216 ]
#  [-4.742708   5.452851 ]
#  [-4.757176   5.4781165]
#  [-4.468812   5.224184 ]
#  [-4.652919   5.317809 ]
#  [-4.699972   5.350121 ]
#  [-4.646487   5.416941 ]
#  [-4.7006307  5.4282484]
#  [-4.7796774  5.4409256]
#  [-4.682902   5.2941995]
#  [-4.7613935  5.4417305]
#  [-4.7368574  5.473317 ]]
#
# Val binnen als de bewoners om hulp vragen, neem het land in, zendt kolonisten naar het gebied of de vorst moet er zelf gaan wonen.
# 1


# RESULTS 2020-05-28 13:17 (locally saved in RoBERTa-002)
#
# RoBERTa  (starting from RobBERT)
# INFO:simpletransformers.classification.classification_model: Training of roberta model complete. Saved to outputs/RoBERTa.
# INFO:simpletransformers.classification.classification_model:{'mcc': 1.0, 'tp': 41, 'tn': 41, 'fp': 0, 'fn': 0, 'eval_loss': 0.005053876175225014}
# [[ 4.6935263  -5.299632  ]
#  [ 4.739876   -5.272859  ]
#  [ 4.804385   -5.299694  ]
#  [ 4.619118   -5.1025896 ]
#  [ 4.7337904  -5.2573957 ]
#  [ 4.29041    -4.804528  ]
#  [ 4.259543   -4.765708  ]
#  [ 3.8679736  -4.427164  ]
#  [ 4.798625   -5.340769  ]
#  [ 4.8995905  -5.435626  ]
#  [ 4.823574   -5.3109994 ]
#  [ 4.847952   -5.3436356 ]
#  [ 4.812998   -5.346259  ]
#  [ 4.7617264  -5.329652  ]
#  [ 4.843856   -5.3378963 ]
#  [-4.935201    5.1497726 ]
#  [-4.798888    5.0554223 ]
#  [-4.7619348   5.063679  ]
#  [-4.7235374   4.8281655 ]
#  [-4.782861    5.0681353 ]
#  [-0.35496756  0.23635131]
#  [-4.4735813   4.5460997 ]
#  [-4.6994047   4.8960485 ]
#  [-4.906059    5.178655  ]
#  [-4.770259    5.063603  ]
#  [-4.7782955   5.0152216 ]
#  [-4.9443974   5.196023  ]
#  [-4.9170933   5.1531954 ]
#  [-5.011503    5.221588  ]
#  [-4.766834    5.0219913 ]
#  [ 4.8645325  -5.4472637 ]
#  [ 4.8867674  -5.3881817 ]
#  [ 4.8108425  -5.4194045 ]
#  [-4.9669847   5.2538815 ]
#  [-4.7485967   5.0354385 ]
#  [-4.954136    5.237783  ]
#  [ 4.8593507  -5.3741894 ]
#  [-4.920395    5.2319417 ]
#  [ 4.929245   -5.416873  ]
#  [-4.9791203   5.2341924 ]
#  [ 4.6392393  -5.2027674 ]
#  [-4.761677    4.95982   ]
#  [ 4.909058   -5.4107447 ]
#  [-4.942155    5.2152033 ]
#  [ 4.6178465  -5.2029457 ]
#  [-4.784238    5.0707846 ]
#  [ 4.899327   -5.4450464 ]
#  [-4.7530656   4.956899  ]
#  [ 4.607891   -5.2004285 ]
#  [-4.7850027   5.0463533 ]
#  [ 4.9210377  -5.475134  ]
#  [-4.739805    4.9209356 ]
#  [ 4.8361053  -5.3827677 ]
#  [ 4.611397   -5.139409  ]
#  [ 4.859295   -5.3458548 ]
#  [ 4.8503084  -5.387227  ]
#  [ 4.8700905  -5.438501  ]
#  [ 4.8844094  -5.4524364 ]
#  [ 4.7202187  -5.316368  ]
#  [ 4.8489137  -5.4567485 ]
#  [ 4.7830987  -5.351675  ]
#  [ 4.464333   -4.9829645 ]
#  [ 4.7363973  -5.2618093 ]
#  [ 4.504025   -5.1156025 ]
#  [ 4.729541   -5.2668085 ]
#  [ 4.8489923  -5.3830385 ]
#  [ 4.9474735  -5.5054145 ]
#  [-4.9825687   5.239164  ]
#  [-5.014144    5.2661014 ]
#  [-4.943488    5.247101  ]
#  [-5.002887    5.261952  ]
#  [-4.9884334   5.2598295 ]
#  [-4.986342    5.263444  ]
#  [-4.941221    5.2461824 ]
#  [-4.949646    5.226528  ]
#  [-4.990345    5.2413616 ]
#  [-4.9485626   5.237907  ]
#  [-4.9887533   5.25799   ]
#  [-5.008194    5.2304497 ]
#  [-4.8797836   5.1673765 ]
#  [-4.9947886   5.2678313 ]
#  [-4.9045987   5.21294   ]]

# RESULTS 2020-05-28 09:35 (locally saved in RoBERTa-001)
#
# RoBERTa  (starting from RobBERT)
#
# INFO:simpletransformers.classification.classification_model: Training of roberta model complete. Saved to outputs/RoBERTa.
# INFO:simpletransformers.classification.classification_model:{'mcc': 1.0, 'tp': 41, 'tn': 41, 'fp': 0, 'fn': 0, 'eval_loss': 0.00381617154330756}
# [[ 4.5753427  -4.6702685 ]
#  [ 4.322606   -4.407047  ]
#  [ 4.3819847  -4.524763  ]
#  [ 3.680751   -3.6428747 ]
#  [ 4.3172317  -4.512207  ]
#  [ 4.032283   -4.213521  ]
#  [ 4.163336   -4.26132   ]
#  [ 0.7128651  -0.27663833]
#  [ 4.5186977  -4.513592  ]
#  [ 4.4825363  -4.6667995 ]
#  [ 4.3640966  -4.5364428 ]
#  [ 4.583468   -4.6392927 ]
#  [ 4.5836935  -4.698412  ]
#  [ 4.501642   -4.477024  ]
#  [ 4.413611   -4.533053  ]
#  [-4.3869634   4.462762  ]
#  [-4.569793    4.9293733 ]
#  [-4.5557084   4.8659973 ]
#  [-4.2262855   4.3428526 ]
#  [-4.5420055   4.930031  ]
#  [-2.1568494   2.4384296 ]
#  [-4.4560766   4.887041  ]
#  [-4.216442    4.4454856 ]
#  [-4.574249    4.711686  ]
#  [-4.572736    4.9549685 ]
#  [-4.5589314   4.902771  ]
#  [-4.4648952   4.575341  ]
#  [-4.422946    4.571241  ]
#  [-4.4544244   4.6230154 ]
#  [-4.5074515   4.8741407 ]
#  [ 4.637774   -4.6137266 ]
#  [ 4.567644   -4.4082522 ]
#  [ 4.608046   -4.5176353 ]
#  [-4.599269    4.8673286 ]
#  [-4.4709277   4.931296  ]
#  [-4.5290375   4.8167367 ]
#  [ 4.623726   -4.6204634 ]
#  [-4.6079974   4.8498473 ]
#  [ 4.6572576  -4.5675945 ]
#  [-4.6000366   4.9142685 ]
#  [ 4.632883   -4.5538635 ]
#  [-4.5706453   4.943189  ]
#  [ 4.62426    -4.5082335 ]
#  [-4.5314255   4.8604674 ]
#  [ 4.633241   -4.536466  ]
#  [-4.5005574   4.9172497 ]
#  [ 4.2877016  -4.0582786 ]
#  [-4.473968    4.8753147 ]
#  [ 4.615284   -4.5457077 ]
#  [-4.515874    4.866644  ]
#  [ 4.289945   -4.0912843 ]
#  [-4.473523    4.907734  ]
#  [ 4.639594   -4.605216  ]
#  [ 4.6339855  -4.5295987 ]
#  [ 4.373646   -4.17642   ]
#  [ 4.6453876  -4.605899  ]
#  [ 4.59813    -4.5983443 ]
#  [ 4.629885   -4.6334314 ]
#  [ 4.599798   -4.506971  ]
#  [ 4.6306686  -4.5693817 ]
#  [ 4.687139   -4.5711784 ]
#  [ 4.4958906  -4.4856234 ]
#  [ 4.6976805  -4.5935545 ]
#  [ 4.6504097  -4.528714  ]
#  [ 4.6792145  -4.451483  ]
#  [ 4.6627192  -4.511315  ]
#  [ 4.45305    -4.4688444 ]
#  [-4.543029    4.8302193 ]
#  [-4.546851    4.8681374 ]
#  [-4.570348    4.8890634 ]
#  [-4.6193366   4.853761  ]
#  [-4.5757713   4.874566  ]
#  [-4.563562    4.8636317 ]
#  [-4.5224524   4.853133  ]
#  [-4.5163317   4.8367414 ]
#  [-4.525481    4.903895  ]
#  [-4.4851637   4.894338  ]
#  [-4.5457535   4.8826365 ]
#  [-4.5538707   4.8580217 ]
#  [-4.513503    4.7983866 ]
#  [-4.5821266   4.859266  ]
#  [-4.574397    4.850874  ]]


# Preparing eval data
eval_data = [
    # zenden
    # Spelling correct
    ["Hij belooft hierbij de Heilige Geest te zenden en geeft ze de opdracht: \"Zoals de Vader Mij heeft uitgezonden, zo zend Ik jullie uit\"", 0],
    ["Enkel Turkse staats tv zendt nog in Koerdisch uit.", 0],
    ["Het leeuwendeel van de ontdekte neutronensterren zendt ook radiostraling uit, inclusief die in röntgen, optisch en gammastraling gedetecteerd zijn", 0],
    ["In Openbaringen 1 schrijft Johannes het volgende: \"Hetgeen gij ziet, schrijf dat in een boek en zend het aan de zeven gemeenten.\"", 0],
    ["RTL 4 zendt het programma in december 2019 en januari 2020 uit.", 0],
    ["Een oplossing is dan een digitale hoofdtelefoon die in en rond de 2,4 GHz ontvangt en zendt.", 0],
    ["En in het heldendicht Hákonarmál is het Hákon de Goede die naar Walhalla wordt gevoerd door de walkure Göndul en Odin zendt Hermóðr en Bragi om hem te begroeten.", 0],
    ["Val binnen als de bewoners om hulp vragen, neem het land in, zend kolonisten naar het gebied of de vorst moet er zelf gaan wonen.", 0],
    ["En mijn broer Aaron is welsprekender dan ik; zend hem als hulp met mij mee om wat ik zeg te bevestigen, want ik ben bang dat zij mij van leugens zullen betichten.", 0],
    ["In een joodse pseudepigrafische tekst, het Testament van Abraham, zendt God de aartsengel Michaël naar Abraham met de boodschap dat deze zich dient voor te bereiden op zijn aanstaande dood.", 0],
    ["Sinds 2013 zendt Groot Nieuws Radio voorafgaand aan Opwekking een Top 100 uit van meest populaire Opwekkingsliederen.", 0],

    ["Via de Post zend ik nog al eens een pakje.", 0],
    ["Sinds wanneer zend jij mij berichten via email?", 0],
    ["Waarom zend je me steeds weer het bos in?", 0],
    ["Die combi-oven zendt zoveel straling uit dat je WiFi er plat van gaat.", 0],

    # With spelling mistake
    ["Hij belooft hierbij de Heilige Geest te zenden en geeft ze de opdracht: \"Zoals de Vader Mij heeft uitgezonden, zo zendt Ik jullie uit\"", 1],
    ["Enkel Turkse staats tv zend nog in Koerdisch uit.", 1],
    ["Het leeuwendeel van de ontdekte neutronensterren zend ook radiostraling uit, inclusief die in röntgen, optisch en gammastraling gedetecteerd zijn", 1],
    ["In Openbaringen 1 schrijft Johannes het volgende: \"Hetgeen gij ziet, schrijf dat in een boek en zendt het aan de zeven gemeenten.\"", 1],
    ["RTL 4 zend het programma in december 2019 en januari 2020 uit.", 1],
    ["Een oplossing is dan een digitale hoofdtelefoon die in en rond de 2,4 GHz ontvangt en zend.", 1],
    ["En in het heldendicht Hákonarmál is het Hákon de Goede die naar Walhalla wordt gevoerd door de walkure Göndul en Odin zend Hermóðr en Bragi om hem te begroeten.", 1],
    ["Val binnen als de bewoners om hulp vragen, neem het land in, zendt kolonisten naar het gebied of de vorst moet er zelf gaan wonen.", 1],
    ["En mijn broer Aaron is welsprekender dan ik; zendt hem als hulp met mij mee om wat ik zeg te bevestigen, want ik ben bang dat zij mij van leugens zullen betichten.", 1],
    ["In een joodse pseudepigrafische tekst, het Testament van Abraham, zend God de aartsengel Michaël naar Abraham met de boodschap dat deze zich dient voor te bereiden op zijn aanstaande dood.", 1],
    ["Sinds 2013 zend Groot Nieuws Radio voorafgaand aan Opwekking een Top 100 uit van meest populaire Opwekkingsliederen.", 1],

    ["Via de Post zendt ik nog al eens een pakje.", 1],
    ["Sinds wanneer zendt jij mij berichten via email?", 1],
    ["Waarom zendt je me steeds weer het bos in?", 1],
    ["Die combi-oven zend zoveel straling uit dat je WiFi er plat van gaat.", 1],

    # worden
    # Spelling correct
    ["Hoe word je gevraagd?", 0],
    ["Wat wordt je gevraagd?", 0],
    ["Wat word je opdringerig, zeg!", 0],

    # With spelling mistake
    ["Hoe wordt je gevraagd?", 1],
    ["Wat word je gevraagd?", 1],
    ["Wat wordt je opdringerig, zeg!", 1],

    # Previous validations (mixed correct and mistakes)
    ["Ik word volgend jaar ook getest.", 0],
    ["Ik wordt helemaal naar hier gehaald.", 1],
    ["Word ik volgend jaar ook uitgenodigd?", 0],
    ["Wordt ik nu al opgeroepen?", 1],

    ["Jij wordt volgend jaar ook getest.", 0],
    ["Jij word helemaal naar hier gehaald.", 1],
    ["Word jij volgend jaar ook uitgenodigd?", 0],
    ["Wordt jij nu al opgeroepen?", 1],

    ["Hij wordt volgend jaar ook getest.", 0],
    ["Hij word helemaal naar hier gehaald.", 1],
    ["Wordt hij volgend jaar ook uitgenodigd?", 0],
    ["Word hij nu al opgeroepen?", 1],

    ["Zij wordt volgend jaar ook getest.", 0],
    ["Zij word helemaal naar hier gehaald.", 1],
    ["Wordt zij volgend jaar ook uitgenodigd?", 0],
    ["Word zij nu al opgeroepen?", 1],

    # Spelling correct
    ["In 1992 gaf de Stichting Popmuseum ook de brochure \"Hoe word je popmuzikant\" uit, met tips voor beginnende popmuzikanten.", 0],
    ["Nou Muijz dan word je bedankt: het werk van maanden sappelen gooi jij met één muisklik weg!", 0],
    ["Voor dat laatste werk kreeg hij kritiek van sommige Vlaams-nationalisten, kritiek die hij afwees met de woorden: \"Ik word door hen zowat beschouwd als een vaandelvluchtige, eenvoudig omdat ik me opsluit binnenshuis en zo hard werk als maar mogelijk is.\"", 0],
    ["Sinds september 2006 presenteert zij voor Talpa de tv-programma's \"Big Brother 6\" en \"Woef: Hoe word ik een beroemde hond?\"", 0],
    ["Op 27 augustus startte \"Hoe word ik een New Yorkse vrouw?\"", 0],
    ["In 2009 presenteerde ze de 4-delige serie \"Hoe word ik een Gooische Vrouw?\"", 0],
    ["Het moet me van het hart dat ik de laatste tijd een trend meen waar te nemen waar ik niet blij van word.", 0],
    ["Twee weken na zijn overlijden verschijnt van Groep Fosko het album 'Van iets maken word je blij'.", 0],
    ["Maar blijkbaar word je alleen beloond voor wat je hebt beloofd en niet op wat je hebt gedaan.", 0],
    ["De auditie kan een selectief karakter hebben: geslaagd of niet, je hebt voldoende talent of niet, of een vergelijkend karakter: je hoort bij de 15 beste kandidaten, dus word je toegelaten, aangezien we er maar 15 toelaten.", 0],
    ["Als je voor een dubbeltje geboren bent, word je nooit een kwartje, lijkt de boodschap aan het eind van de film.", 0],
    ["Simon Carmiggelt noteert in een van zijn cursiefjes: \"We kunnen geestig zijn in Amsterdam, daar word je weleens beroerd van.\"", 0],
    ["Ook de voortdurend terugkerende vaststelling dat wikipedia voor universitair studenten en wetenschappers nooit een gezaghebbende bron zal zijn, word ik een beetje zat.", 0],
    ["Dan word ik opgeofferd aan het ego van degene die een verkeerde beslissing heeft genomen, en dat lijkt me niet terecht.", 0],
    ["Ik word gewoon het offer dat gebracht moet worden om jullie te legitimeren een jacobijns schrikbewind te vestigen.", 0],

    # With spelling mistake
    ["In 1992 gaf de Stichting Popmuseum ook de brochure \"Hoe wordt je popmuzikant\" uit, met tips voor beginnende popmuzikanten.", 1],
    ["Nou Muijz dan wordt je bedankt: het werk van maanden sappelen gooi jij met één muisklik weg!", 1],
    ["Voor dat laatste werk kreeg hij kritiek van sommige Vlaams-nationalisten, kritiek die hij afwees met de woorden: \"Ik wordt door hen zowat beschouwd als een vaandelvluchtige, eenvoudig omdat ik me opsluit binnenshuis en zo hard werk als maar mogelijk is.\"", 1],
    ["Sinds september 2006 presenteert zij voor Talpa de tv-programma's \"Big Brother 6\" en \"Woef: Hoe wordt ik een beroemde hond?\"", 1],
    ["Op 27 augustus startte \"Hoe wordt ik een New Yorkse vrouw?\"", 1],
    ["In 2009 presenteerde ze de 4-delige serie \"Hoe wordt ik een Gooische Vrouw?\"", 1],
    ["Het moet me van het hart dat ik de laatste tijd een trend meen waar te nemen waar ik niet blij van wordt.", 1],
    ["Twee weken na zijn overlijden verschijnt van Groep Fosko het album 'Van iets maken wordt je blij'.", 1],
    ["Maar blijkbaar wordt je alleen beloond voor wat je hebt beloofd en niet op wat je hebt gedaan.", 1],
    ["De auditie kan een selectief karakter hebben: geslaagd of niet, je hebt voldoende talent of niet, of een vergelijkend karakter: je hoort bij de 15 beste kandidaten, dus wordt je toegelaten, aangezien we er maar 15 toelaten.", 1],
    ["Als je voor een dubbeltje geboren bent, wordt je nooit een kwartje, lijkt de boodschap aan het eind van de film.", 1],
    ["Simon Carmiggelt noteert in een van zijn cursiefjes: \"We kunnen geestig zijn in Amsterdam, daar wordt je weleens beroerd van.\"", 1],
    ["Ook de voortdurend terugkerende vaststelling dat wikipedia voor universitair studenten en wetenschappers nooit een gezaghebbende bron zal zijn, wordt ik een beetje zat.", 1],
    ["Dan wordt ik opgeofferd aan het ego van degene die een verkeerde beslissing heeft genomen, en dat lijkt me niet terecht.", 1],
    ["Ik wordt gewoon het offer dat gebracht moet worden om jullie te legitimeren een jacobijns schrikbewind te vestigen.", 1],

    # Vinden and lopen
    # Spelling correct
    ["Ik vind dit toch niet zo mooi.", 0],
    ["Wat vind jij van al die aandacht?", 0],
    ["Hoe vind ik nu de ingang?", 0],
    ["Het is toch erg dat hij dat niet vindt.", 0],
    ["Hoe zwaar vindt hij de opleiding?", 0],

    ["Ik loop er zo maar voorbij.", 0],
    ["Loop jij ook zo snel?", 0],
    ["Je loopt daar beter niet telkens over.", 0],
    ["Jij loopt echt helemaal naar zee?", 0],
    ["En daarom loopt hij er met een grote bocht omheen.", 0],

    # With spelling mistake
    ["Ik vindt dit toch niet zo mooi.", 1],
    ["Wat vindt jij van al die aandacht?", 1],
    ["Hoe vindt ik nu de ingang?", 1],
    ["Het is toch erg dat hij dat niet vind.", 1], # Incorrectly evaluated to "0"
    ["Hoe zwaar vind hij de opleiding?", 1],

    ["Ik loopt er zo maar voorbij.", 1],
    ["Loopt jij ook zo snel?", 1],
    ["Je loop daar beter niet telkens over.", 1],
    ["Jij loop echt helemaal naar zee?", 1],
    ["En daarom loop hij er met een grote bocht omheen.", 1],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]


# Preparing train data
# Correct Dutch sentences have a '0' label
# Sentences with a "dt" mistake for worden (word vs. wordt) and zenden (zend vs. zendt) have a '1' label
train_data = [
    # zenden
    # Correct spelling; synthetic
    ["Ik zend een pakje.", 0],
    ["Ik zend je een pakje.", 0],
    ["Ik zend u een heel mooie trui.", 0],
    ["Zend ik u best morgen al de nieuwe catalogus?", 0],
    ["Waarom zend ik je niet zelf voor die moeilijke opdracht?", 0],
    ["Wat zend ik best naar mijn moeder voor haar verjaardag?", 0],
    ["Zend ik die bestelling op donderdag op, en toch komt die pas op dinsdag aan!", 0],
    ["Je zendt het toch nog vandaag op, hoop ik.", 0],
    ["Jij zendt een hele grote doos naar oma.", 0],
    ["Zend je dat straks uit?", 0],
    ["Zend je deze bestelling nog even naar de klant?", 0],
    ["Zend je ook nog die laatste versie even door?", 0],
    ["Zend je dat vanavond laat ook uit?", 0],
    ["Waarom zend je dat niet via onze koerier?", 0],
    ["Zend jij echt die fiets via de post?", 0],
    ["Welke baas zendt je nu helemaal naar London voor 1 klant?", 0],
    ["Hoe laat zendt u de de nieuwe versie door?", 0],
    ["Zendt u dit volgende week al uit?", 0],

    # Correct spelling; from nl.wikipedia (in _most_ cases, I needed to correct "zend" to "zendt" for 3rd person singular usage, except citations from the bible)
    ["Een gebed van Apollonius van Tyana uit ongeveer het jaar 23 AD: \"O, God van de Zon, zend me zover rond de wereld als goed is voor mij en jou, en dat ik goede mensen mag ontmoeten, maar nooit de slechte leer kennen, noch zij mij.", 0],
    ["Voor het ontvangen van omroepsignalen wordt de kenmerkende zend- en ontvangstmast van Naaldwijk gebouwd.", 0],
    ["Met de wet van behoud van energie geldt dan ook dat met 100 W aan de zend-trap nog steeds minder dan 100 W door de antenne uitgestraald zal worden.", 0],
    ["Deze fabriek richtte zich in eerste instantie op de productie van zend- en ontvanginstallaties voor schepen en vliegtuigen.", 0],
    ["Met zijn zend- en ontvangschoen roept hij Jerom en professor Barabas op en zij verslaan de tempelwachters in de doolhof.", 0],
    ["Het woord radio wordt eveneens gebruikt als afkorting voor radio-omroep, radio-ontvanger en zend- en ontvangapparatuur.", 0],
    ["Sponsor Motorola is een Amerikaans elektronicabedrijf en het experimenteerde als eerste met het gebruik van zend- en ontvangstapparatuur in de koers.", 0],
    ["De toren is ontworpen als onderkomen voor omroepzenders en als zend- en ontvangstation voor PTT/KPN straalverbindingen voor telefonie.", 0],
    ["En zend daarna een evaluatieteam in als iedereen is afgekoeld.", 0],
    ["Via zijn eigen zend-installatie zond hij berichten uit naar alle hoeken van de wereld; Amerika, Groenland, Indië en Australië, waarbij hij zich vlot van verschillende vreemde talen bediende.", 0],
    ["Het blad bevat naast verenigingsnieuws ook technische artikelen over zend- en ontvangtechnieken.", 0],
    ["Ten tijde van de kabel van 1866 waren zowel de kabelfabricage als de zend- en ontvangstapparatuur aanzienlijk verbeterd.", 0],
    ["Hiermee neemt de vaste autotelefoon de telefoonfunctie van de mobiele telefoon over: De zend-ontvang functie van de mobiele telefoon wordt in een stand-by modus gezet.", 0],
    ["Het voordeel van deze rSAP autotelefoons is een veel betere zend- en ontvangstkwaliteit.", 0],
    ["Ook de zend- en ontvangstkwaliteit is beter bij rsap.", 0],
    ["Sinds 2013 zendt ook 2BE de serie uit.", 0],
    ["In december 2010 zendt televisie- en internet aanbieder Ziggo een tweetal nieuwe 2 Meter Sessies uitzendingen uit op haar digitale televisiekanaal.", 0],
    ["Dat moge blijken uit het feit dat voor zowel zend- als ontvangstantenne in de UHF-omroepband kanalen 21-69 vrijwel altijd een universeel antennetype gebruikt wordt.", 0],
    ["'s Morgens zendt het kanaal de programmering van 24 h uit, met nieuws en achtergronden.", 0],
    ["Sinds 15 december 2008 zendt de zender uit tussen 6 en 20.15 uur op hetzelfde kanaal als Comedy Central Duitsland, en dit ook via de satelliet.", 0],
    ["Hierbij zendt de gebruiker alleen mentale energie naar zijn opponent, die daar wel door wordt overheerst en niet meer uit eigen wil kan handelen zolang deze Jutsu bezig is.", 0],
    ["Sinds 2017 zendt Veronica kwalificatiewedstrijden van het Nederlands vrouwenvoetbalelftal uit.", 0],
    ["De zend- en ontvangstinstallaties daarvoor waren echter groot en moeilijk verplaatsbaar.", 0],
    ["Zie, Ik zend u de profeet Elia, voordat de grote en geduchte dag van Jahweh komt.", 0],
    ["Het woord radio wordt eveneens gebruikt als afkorting voor radio-omroep, radio-ontvanger en zend- en ontvangapparatuur.", 0],
    ["Behalve om rechtstreeks tussen twee zend-ontvangers met elkaar te communiceren, is D-star uitermate geschikt om ook via een D-star repeater of een D-star hotspot te communiceren.", 0],
    ["Een Repeater is een zend-ontvanger welke onbemand het ontvangen radiosignaal op een andere frequentie her-uitzendt.", 0],
    ["BBC Radio 1 richt zich vooral op de doelgroep 15 tot 29 jaar, en zendt vooral pop, elektronische, alternatieve en rock muziek uit.", 0],
    ["Een antenne-array is een samenstel van een aantal zend- of ontvangstantennes, om voor een bepaalde frequentie een optimale energieoverdracht in  één of meer richtingen te bewerkstelligen.", 0],
    ["Momenteel zendt Boomerang de tekenfilm uit, geheel opnieuw ingesproken, maar met dezelfde stemacteurs.", 0],
    ["In het midden- en noorden van Schotland zendt STV uit, in Noord-Ierland zendt UTV uit.", 0],
    ["Dit onderdeel zend een schokgolf uit waarmee je zwakke muren kan openbreken.", 0],
    ["In 1917 stonden er al tijdelijke zend- en ontvangststations voor draadloze telegrafie op de lange golf op de hoogvlakte Malabar nabij Bandoeng op het Nederlands-Indische eiland Java, voor contact met het moederland.", 0],
    ["Deze zender zendt uit in het Engels en is gericht op de islam.", 0],
    ["Het kanaal zendt dagelijks uit vanuit zijn studio's op Stamford Bridge.", 0],
    ["Namelijk door zowel aan de zend- als aan de ontvangstzijde kathodestraalbuizen te gebruiken.", 0],
    ["In de daarop volgende jaren breidde hij zijn draadloze netwerk uit door wereldwijd diverse zend- en ontvangststations te bouwen.", 0],
    ["Tijdens een vuurgevecht met de Duitsers waren ze hun uitrusting, wapens en radio zend-ontvanger kwijt geraakt.", 0],
    ["Wanneer Big Brother wordt uitgezonden op Channel 4, zendt E4 veel extra programma's uit rond de serie.", 0],
    ["De Church of Scotland besluit hen te gaan sponsoren en zendt het echtpaar naar Chamba, een leprakolonie aan de voet van de Himalaya.", 0],
    ["De dieven schrikken en rennen weg en Salam pakt het geld en zendt een gedeelte naar zijn meester.", 0],
    ["Marinus Vader ontving inmiddels een zend-ontvanger via route Zwaantje.", 0],
    ["Helaas zendt geen 1 kanaal constant uit 70% van de tijd andere zaken.", 0],
    ["De zender zendt 24 uur per dag programma's uit.", 0],
    ["Aanschouw, Heer, de droefheid van Uw volk, en zend ons degene, die U zenden wil.", 0],
    ["Wanneer in februari 1799 nog geen antwoord uit Nederland is gekomen, zendt men een tweede verzoek, vergezeld van een bedrag van f 1100,--.", 0],
    ["Sindsdien zendt het radiostation 24 uur per dag, zeven dagen in de week uit.", 0],
    ["Korte tijd daarna zendt de hertog een korte brief waarin hij stelt dat Bredevoort bij de graafschap Zutphen hoort en aan de voorouders van Arnold in pand is gegeven.", 0],
    ["Midland FM is de eerste en oudste streekomroep van Nederland en zendt uit in de gemeenten Renswoude, Scherpenzeel, Veenendaal, Woudenberg.", 0],
    ["Zoals de meeste planetaire nevels zendt de halternevel zijn zichtbare licht voornamelijk in één enkele spectraallijn uit, 500,7 nm.", 0],
    ["Daarnaast zou hij voor allen de zend- en ontvangstapparatuur, de verrekijkers, de codekaarten, hun bewapening en hun privéspullen beheren.", 0],

    ["Verder zendt het netwerk nog uit via drie televisiekanalen die niet in Telefe’s beheer zijn.", 0],
    ["Het schip zendt signalen en bevelen uit naar de battle droids, droidekas en droid starfighters die in de buurt actief zijn.", 0],
    ["Sinds 12 december 2016 zendt Spike dagelijks uit van 21.05 tot 5.00 uur op het kanaal van Nickelodeon.", 0],
    ["In vergelijking met de Nederlandse versie zendt de zender niet 24 uur per dag uit, en dat terwijl het sinds 12 december 2016 in Nederland wel het geval is.", 0],
    ["Het schip zendt een S.O.S. uit en het eerste schot van de duikboot verbrijzelt het roer.", 0],
    ["In het budget van de zone voor 2017 wordt 620.000 euro geïnvesteerd in de aankoop van persoonlijke beschermingsmiddelen, 385.000 euro in de vernieuwing van zend- en oproepapparatuur, 2,5 miljoen euro in nieuw materieel en wagens 440.000 euro in opleidingen.", 0],
    ["Het betreft een massive MIMO-installatie van Huawei met meerdere kleine zend- en ontvangstantennes, geplaatst op het Leidseplein door T-Mobile.", 0],
    ["Het radiostation zendt van 5 's ochtends tot 4 uur 's middags en van 5 uur 's middags tot 10 uur 's avonds uit.", 0],
    ["Naast de reguliere programma's zendt Omroep Helmond ook regelmatig uit vanaf locatie, als er ergens in Helmond een bijzonder evenement is.", 0],
    ["In alle Wolf’s kan zend- en ontvangstapparatuur worden geplaatst.", 0],
    ["Echter tijdens de Tweede Wereldoorlog heeft men nog onvoldoende vertrouwen in de apparatuur en zendt men met vluchten nog twee duiven mee.", 0],
    ["Zend, de heilige taal van de Zoroastriërs zou van dit Sanzar afstammen.", 0],
    ["Dit biecht zij op aan haar verloofde Wamgans, waarop deze Sofrelli tijdens de bruiloft weg zendt.", 0],

    # Synthetic data, close to failing evaluations for zend
    ["In dat boek, schrijft Jan het volgende: \"Hetgeen gij hoort, onthoud dat en zend het naar je broers.\"", 0],
    ["Zend zo duidelijke mogelijke instructies naar alle deelnemers!", 0],
    ["Ga er direct naartoe en zend hulp naar de getroffen zones!", 0],
    ["Vraag het even na en zend dan direct het resultaat door.", 0],
    ["Wanneer je bent aangekomen, zend je me dan direct een SMS?", 0],
    ["En Hij zei: \"Zend vele groeten naar alle nieuwe leden.\"", 0],
    ["Toen gaf ze de opdracht: \"Verzamel modern materiaal en zend het per direct naar onze partners\"", 0],
    ["Met die puntkomma erbij is hij veel sterker dan ik; zend hem alvast mijn felicitaties!", 0],
    ["Door die puntkomma is hij ook veel groter dan ik; zend hem een helm!", 0],
    ["Die antenne zendt zo'n zwak signaal dat je hem nauwelijks kan ontvangen", 0],
    ["Radio Scorpio zendt in FM op 106.0 MHz", 0],
    ["Die ster zendt ook Gamma straling de ruimte in, maar ik ben niet zeker als dat wel kan kloppen.", 0],
    ["Ondanks alles, zendt het toch een heel sterk signaal.", 0],
    ["In het nieuwe Testament, zendt God de engelen allemaal naar boven.", 0],
    ["In het Testament van Jonas, zendt God de aartsengel Michaël naar Abraham met de boodschap dat het einde nadert.", 0],
    ["God zendt zijn dochters uit.", 0],
    ["God zendt het meeste pakjes, op Sinterklaas na.", 0],

    # With spelling mistake
    ["Ik zendt een pakje.", 1],
    ["Ik zendt je een pakje.", 1],
    ["Ik zendt u een heel mooie trui.", 1],
    ["Zendt ik u best morgen al de nieuwe catalogus?", 1],
    ["Waarom zendt ik je niet zelf voor die moeilijke opdracht?", 1],
    ["Wat zendt ik best naar mijn moeder voor haar verjaardag?", 1],
    ["Zendt ik die bestelling op donderdag op, en toch komt die pas op dinsdag aan!", 1],
    ["Je zend het toch nog vandaag op, hoop ik.", 1],
    ["Jij zend een hele grote doos naar oma.", 1],
    ["Zendt je dat straks uit?", 1],
    ["Zendt je deze bestelling nog even naar de klant?", 1],
    ["Zendt je ook nog die laatste versie even door?", 1],
    ["Zendt je dat vanavond laat ook uit?", 1],
    ["Waarom zendt je dat niet via onze koerier?", 1],
    ["Zendt jij echt die fiets via de post?", 1],
    ["Welke baas zend je nu helemaal naar London voor 1 klant?", 1],
    ["Hoe laat zend u de de nieuwe versie door?", 1],
    ["Zend u dit volgende week al uit?", 1],

    # With spelling mistake; from nl.wikipedia (corrected in many places)
    ["Een gebed van Apollonius van Tyana uit ongeveer het jaar 23 AD: \"O, God van de Zon, zendt me zover rond de wereld als goed is voor mij en jou, en dat ik goede mensen mag ontmoeten, maar nooit de slechte leer kennen, noch zij mij.", 1],
    ["En zendt daarna een evaluatieteam in als iedereen is afgekoeld.", 1],
    ["Sinds 2013 zend ook 2BE de serie uit.", 1],
    ["In december 2010 zend televisie- en internet aanbieder Ziggo een tweetal nieuwe 2 Meter Sessies uitzendingen uit op haar digitale televisiekanaal.", 1],
    ["'s Morgens zend het kanaal de programmering van 24 h uit, met nieuws en achtergronden.", 1],
    ["Sinds 15 december 2008 zend de zender uit tussen 6 en 20.15 uur op hetzelfde kanaal als Comedy Central Duitsland, en dit ook via de satelliet.", 1],
    ["Hierbij zend de gebruiker alleen mentale energie naar zijn opponent, die daar wel door wordt overheerst en niet meer uit eigen wil kan handelen zolang deze Jutsu bezig is.", 1],
    ["Sinds 2017 zend Veronica kwalificatiewedstrijden van het Nederlands vrouwenvoetbalelftal uit.", 1],
    ["Zie, Ik zendt u de profeet Elia, voordat de grote en geduchte dag van Jahweh komt.", 1],
    ["BBC Radio 1 richt zich vooral op de doelgroep 15 tot 29 jaar, en zend vooral pop, elektronische, alternatieve en rock muziek uit.", 1],
    ["Momenteel zend Boomerang de tekenfilm uit, geheel opnieuw ingesproken, maar met dezelfde stemacteurs.", 1],
    ["In het midden- en noorden van Schotland zend STV uit, in Noord-Ierland zend UTV uit.", 1],
    ["Dit onderdeel zendt een schokgolf uit waarmee je zwakke muren kan openbreken.", 1],
    ["Deze zender zend uit in het Engels en is gericht op de islam.", 1],
    ["Het kanaal zend dagelijks uit vanuit zijn studio's op Stamford Bridge.", 1],
    ["Wanneer Big Brother wordt uitgezonden op Channel 4, zend E4 veel extra programma's uit rond de serie.", 1],
    ["De Church of Scotland besluit hen te gaan sponsoren en zend het echtpaar naar Chamba, een leprakolonie aan de voet van de Himalaya.", 1],
    ["De dieven schrikken en rennen weg en Salam pakt het geld en zend een gedeelte naar zijn meester.", 1],
    ["Helaas zend geen 1 kanaal constant uit 70% van de tijd andere zaken.", 1],
    ["De zender zend 24 uur per dag programma's uit.", 1],
    ["Aanschouw, Heer, de droefheid van Uw volk, en zendt ons degene, die U zenden wil.", 1],
    ["Wanneer in februari 1799 nog geen antwoord uit Nederland is gekomen, zend men een tweede verzoek, vergezeld van een bedrag van f 1100,--.", 1],
    ["Sindsdien zend het radiostation 24 uur per dag, zeven dagen in de week uit.", 1],
    ["Korte tijd daarna zend de hertog een korte brief waarin hij stelt dat Bredevoort bij de graafschap Zutphen hoort en aan de voorouders van Arnold in pand is gegeven.", 1],
    ["Midland FM is de eerste en oudste streekomroep van Nederland en zend uit in de gemeenten Renswoude, Scherpenzeel, Veenendaal, Woudenberg.", 1],
    ["Zoals de meeste planetaire nevels zend de halternevel zijn zichtbare licht voornamelijk in één enkele spectraallijn uit, 500,7 nm.", 1],

    ["Verder zend het netwerk nog uit via drie televisiekanalen die niet in Telefe’s beheer zijn.", 1],
    ["Het schip zend signalen en bevelen uit naar de battle droids, droidekas en droid starfighters die in de buurt actief zijn.", 1],
    ["Sinds 12 december 2016 zend Spike dagelijks uit van 21.05 tot 5.00 uur op het kanaal van Nickelodeon.", 1],
    ["In vergelijking met de Nederlandse versie zend de zender niet 24 uur per dag uit, en dat terwijl het sinds 12 december 2016 in Nederland wel het geval is.", 1],
    ["Het schip zend een S.O.S. uit en het eerste schot van de duikboot verbrijzelt het roer.", 1],
    ["Het betreft een massive MIMO-installatie van Huawei met meerdere kleine zend- en ontvangstantennes, geplaatst op het Leidseplein door T-Mobile.", 1],
    ["Het radiostation zend van 5 's ochtends tot 4 uur 's middags en van 5 uur 's middags tot 10 uur 's avonds uit.", 1],
    ["Naast de reguliere programma's zend Omroep Helmond ook regelmatig uit vanaf locatie, als er ergens in Helmond een bijzonder evenement is.", 1],
    ["Echter tijdens de Tweede Wereldoorlog heeft men nog onvoldoende vertrouwen in de apparatuur en zend men met vluchten nog twee duiven mee.", 1],
    ["Dit biecht zij op aan haar verloofde Wamgans, waarop deze Sofrelli tijdens de bruiloft weg zend.", 1],

    # Synthetic data, close to failing evaluations for zend
    ["In dat boek, schrijft Jan het volgende: \"Hetgeen gij hoort, onthoud dat en zendt het naar je broers.\"", 1],
    ["Zendt zo duidelijke mogelijke instructies naar alle deelnemers!", 1],
    ["Ga er direct naartoe en zendt hulp naar de getroffen zones!", 1],
    ["Vraag het even na en zendt dan direct het resultaat door.", 1],
    ["Wanneer je bent aangekomen, zendt je me dan direct een SMS?", 1],
    ["En Hij zei: \"Zendt vele groeten naar alle nieuwe leden.\"", 1],
    ["Toen gaf ze de opdracht: \"Verzamel modern materiaal en zendt het per direct naar onze partners\"", 1],
    ["Met die puntkomma erbij is hij veel sterker dan ik; zendt hem alvast mijn felicitaties!", 1],
    ["Door die puntkomma is hij ook veel groter dan ik; zendt hem een helm!", 1],
    ["Die antenne zend zo'n zwak signaal dat je hem nauwelijks kan ontvangen", 1],
    ["Radio Scorpio zend in FM op 106.0 MHz", 1],
    ["Die ster zend ook Gamma straling de ruimte in, maar ik ben niet zeker als dat wel kan kloppen.", 1],
    ["Ondanks alles, zend het toch een heel sterk signaal.", 1],
    ["In het nieuwe Testament, zend God de engelen allemaal naar boven.", 1],
    ["In het Testament van Jonas, zend God de aartsengel Michaël naar Abraham met de boodschap dat het einde nadert.", 1],
    ["God zend zijn dochters uit.", 1],
    ["God zend het meeste pakjes, op Sinterklaas na.", 1],

    # worden
    # Correct
    ["Ik word is zonder t", 0],
    ["Ik word warm.", 0],
    ["Ik word enthousiast.", 0],
    ["Ik word eigenaar van een nieuwe kat.", 0],
    ["Hierdoor word ik ook helemaal verrast.", 0],
    ["Hoe word ik eigenlijk geholpen?", 0],
    ["Waarom word ik van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder word ik hierdoor afgezet?", 0],

    # With spelling mistake
    ["Ik wordt is zonder t", 1],
    ["Ik wordt warm.", 1],
    ["Ik wordt enthousiast.", 1],
    ["Ik wordt eigenaar van een nieuwe kat.", 1],
    ["Hierdoor wordt ik ook helemaal verrast.", 1],
    ["Hoe wordt ik eigenlijk geholpen?", 1],
    ["Waarom wordt ik van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder wordt ik hierdoor afgezet?", 1],

    # Correct
    ["Jij wordt is met t", 0],
    ["Jij wordt warm.", 0],
    ["Jij wordt enthousiast.", 0],
    ["Jij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor word jij ook helemaal verrast.", 0],
    ["Hoe word jij eigenlijk geholpen?", 0],
    ["Waarom word jij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder word jij hierdoor afgezet?", 0],

    # With spelling mistake
    ["Jij word is met t", 1],
    ["Jij word warm.", 1],
    ["Jij word enthousiast.", 1],
    ["Jij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor wordt jij ook helemaal verrast.", 1],
    ["Hoe wordt jij eigenlijk geholpen?", 1],
    ["Waarom wordt jij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder wordt jij hierdoor afgezet?", 1],

    # Correct
    ["Hij wordt is met t", 0],
    ["Hij wordt warm.", 0],
    ["Hij wordt enthousiast.", 0],
    ["Hij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor wordt hij ook helemaal verrast.", 0],
    ["Hoe wordt hij eigenlijk geholpen?", 0],
    ["Waarom wordt hij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder wordt hij hierdoor afgezet?", 0],

    # With spelling mistake
    ["Hij word is met t", 1],
    ["Hij word warm.", 1],
    ["Hij word enthousiast.", 1],
    ["Hij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor word hij ook helemaal verrast.", 1],
    ["Hoe word hij eigenlijk geholpen?", 1],
    ["Waarom word hij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder word hij hierdoor afgezet?", 1],

    # Correct
    ["Zij wordt is met t", 0],
    ["Zij wordt warm.", 0],
    ["Zij wordt enthousiast.", 0],
    ["Zij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor wordt zij ook helemaal verrast.", 0],
    ["Hoe wordt zij eigenlijk geholpen?", 0],
    ["Waarom wordt zij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder wordt zij hierdoor afgezet?", 0],

    # With spelling mistake
    ["Zij word is met t", 1],
    ["Zij word warm.", 1],
    ["Zij word enthousiast.", 1],
    ["Zij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor word zij ook helemaal verrast.", 1],
    ["Hoe word zij eigenlijk geholpen?", 1],
    ["Waarom word zij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder word zij hierdoor afgezet?", 1],

    # Correct "je" is _not_ subject
    ["Wat wordt je verweten?", 0],
    ["Wat wordt je aangedaan?", 0],
    ["Wat wordt je gegeven?", 0],
    ["Wat wordt je geschonken?", 0],
    ["Wat wordt er je verweten?", 0],
    ["Wat wordt er je gevraagd?", 0],
    ["Wat wordt er je nu aangedaan?", 0],
    ["Wat wordt er je gegeven?", 0],
    ["Wat wordt er je geschonken?", 0],
    ["Het pakje wordt je morgen geleverd.", 0],
    ["Het contract wordt je overmorgen verstuurd.", 0],
    ["Je geschenk wordt je zo snel mogelijk toegezonden.", 0],

    ["Wat wordt jou verweten?", 0],
    ["Wat wordt jou aangedaan?", 0],
    ["Wat wordt jou gegeven?", 0],
    ["Wat wordt jou geschonken?", 0],
    ["Wat wordt er jou verweten?", 0],
    ["Wat wordt er jou gevraagd?", 0],
    ["Wat wordt er jou nu aangedaan?", 0],
    ["Wat wordt er jou gegeven?", 0],

    # Correct "je" is the subject
    ["Wat word je groot, zeg!", 0],
    ["Waarom word je in dat dossier vervolgd?", 0],
    ["Waarom word je gevolgd?", 0],
    ["Hoe word je gevolgd?", 0],
    ["Waarom word je zo zwaar bekritiseerd?", 0],
    ["Waarom word je zo behandeld?", 0],
    ["Wat word jij toch geliefd!", 0],
    ["Wat word jij toch snel groot!", 0],
    ["Wat word jij later?", 0],
    ["Hoe word je in dat dossier vervolgd?", 0],
    ["Hoe word je gevolgd?", 0],
    ["In welk artikel word je zo zwaar bekritiseerd?", 0],
    ["Waarom word je zo behandeld?", 0],

    # With spelling mistake, "je" is _not_ subject
    ["Wat word je aangedaan?", 1],
    ["Wat word je gegeven?", 1],
    ["Wat word je geschonken?", 1],
    ["Wat word er je verweten?", 1],
    ["Wat word er je gevraagd?", 1],
    ["Wat word er je nu aangedaan?", 1],
    ["Wat word er je gegeven?", 1],
    ["Wat word er je geschonken?", 1],
    ["Het pakje word je morgen geleverd.", 1],
    ["Het contract word je overmorgen verstuurd.", 1],
    ["Je geschenk word je zo snel mogelijk toegezonden.", 1],

    ["Wat word jou verweten?", 1],
    ["Wat word jou aangedaan?", 1],
    ["Wat word jou gegeven?", 1],
    ["Wat word jou geschonken?", 1],
    ["Wat word er jou verweten?", 1],
    ["Wat word er jou gevraagd?", 1],
    ["Wat word er jou nu aangedaan?", 1],
    ["Wat word er jou gegeven?", 1],

    # With spelling mistake, "je" is the subject
    ["Wat wordt je groot, zeg!", 1],
    ["Waarom wordt je in dat dossier vervolgd?", 1],
    ["Waarom wordt je gevolgd?", 1],
    ["Hoe wordt je gevolgd?", 1],
    ["Waarom wordt je zo zwaar bekritiseerd?", 1],
    ["Waarom wordt je zo behandeld?", 1],
    ["Wat wordt jij toch geliefd!", 1],
    ["Wat wordt jij toch snel groot!", 1],
    ["Wat wordt jij later?", 1],
    ["Hoe wordt je in dat dossier vervolgd?", 1],
    ["Hoe wordt je gevolgd?", 1],
    ["In welk artikel wordt je zo zwaar bekritiseerd?", 1],
    ["Waarom wordt je zo behandeld?", 1],

    # Correct values for 'word'
    ["Daar word ik hartstikke gek van.", 0],
    ["Ik hoop dat dit dus niet bij jou gedaan word.", 0],
    ["Zo verving von Baader Descartes' cogito ergo sum door cogitor ergo sum: ik word gedacht, dus ik ben.", 0],
    ["Ik ga door tot Parijs en zal winnen, tenzij ik vermoord word.", 0],
    ["Hij zou op de dag van de verkiezing tegen zijn moeder gezegd hebben Of ik keer terug als pontiflex, of ik word voor altijd verbannen.", 0],
    ["Rond het thema van de Vlaamse onafhankelijkheid schreef hij ‘Operatie Vlaamse Onafhankelijkheid‘, een verslag van de evaluatiedag ‘Volk, word staat’ dat op 24 november 2007 plaatsvond in het Federale Parlement.", 0],
    ["Als je goed leeft, kun je na de dood in een hogere kaste terecht komen en uiteindelijk word je bevrijd van het aardse leven om samen te leven met god.", 0],
    ["Vertaling: Ik word beschermd of hij beschermt mij.", 0],
    ["Als je van deze appels eet, word je onsterfelijk.", 0],
    ["In 2008 begon BNN met een grootse campagne onder het motto 'Koop dit, word lid'.", 0],
    ["Nu pas word ik me ervan bewust dat mijn werk in zwart, wit en kleine kleurvlakken alleen maar 'tekenen' in olieverf is geweest.", 0],
    ["Wel liet hij ooit in een interview optekenen: Oud met een d word ik niet waarschijnlijk, gezien de grote hoeveelheid koffie en sigaretten die ik consumeer, en de geringe hoeveelheid slaap die ik geniet.", 0],
    ["In 2008 verscheen het eerste luisterboek van doctorandus P, waarop hij zelf drie eigen verhalen voorleest: Sven de Bevrijder, Ik word vermoord en Tiens, Tiens, afgewisseld met een aantal ollekebollekes.", 0],
    ["Juridisch word je bij een koop dus nog geen eigenaar!", 0],
    ["Zuiderspel word georganiseerd in Hotel and Congrescentrum Koningshof te Veldhoven.", 0],
    ["Nescio's bevlogen proza wordt in de tweede alinea komisch-ruw onderbroken door Bavink: Toen zei Bavink: 'Ik word een beroemd man,' zooals een ander zou zeggen: 'Ze hebben me een dubbeltje te veel afgezet,' en we voelden ons bekocht, alle drie, Bavink, Bekker en ik.", 0],
    ["In het West-Vlaams zegt men dus niet ik word ziek, maar men zegt ik kom ziek.", 0],
    ["Vroeg of laat word ik natuurlijk vergeten, zei Havel bij die gelegenheid, maar gelukkig is er nog altijd die foto waarop ik met Arnold Schwarzenegger sta.", 0],
    ["Want 'stel je je kwetsbaar op, dan word je gekwetst', aldus Cruijff.", 0],
    ["Daar word je met al je idealen natuurlijk ontzettend agressief van, maar er is nu eenmaal geen vooropleiding voor wethouder.", 0],
    ["Na de uitspraak \"Ik word daar zo moe van\" valt hij meestal direct in slaap.", 0],
    ["Dachau heeft een blijvend stempel op hem gedrukt: Banger word ik voor mijn eigen wezen, Dachau schoof een raster voor mijn ziel en wie daarin opgenomen is geweest, zal de dood tot zijn dood met zich meedragen.", 0],
    ["Op zijn sterfbed sprak hij de woorden uit: Aan mijn geleerdheid heb ik nu niets meer; mijn dogmatiek baat mij ook nu niet meer; alleen door het geloof word ik zalig.", 0],
    ["Römer vond dat mensen daar behandeld worden als gebruiksvoorwerpen: Eerst word je in de watten gelegd en later laten ze je net zo hard weer vallen.", 0],
    ["Ik word nog meer gehaat dan voorheen, omdat ik nu goed bij kas ben.", 0],
    ["Onderdelen van de set zoals de kippen en bloemen werden meegegeven aan het publiek en verloot onder mensen die lid waren geworden van BNN tijdens de “Koop dit, word lid”-actie.", 0],
    ["Ik word misschien nog eens medeplichtig gehouden, als Mitja zich aan uw vader mocht vergrijpen.", 0],
    ["Om de nieuwe SIMD-instructies te kunnen implementeren heeft Intel 4 nieuwe datatypen ingevoerd, de packed versies van de byte, word, doubleword en quadword.", 0],
    ["Natuurlijk ben je lid van een familie door geboorte en word je op een bepaalde leeftijd ingewijd, door middel van rituelen en een eed voor het leven, maar je kunt ook bij een familie komen als je door de familie wordt geaccepteerd en ritueel wordt omgedoopt.", 0],
    ["Houd er ook rekening mee dat de spellingcontrole wel werkwoordsvormen die echt fout zijn markeert, zoals heeft gebruikd, maar niet de werkwoordsvormen waarin wel goede woorden staan maar die grammaticaal niet correct zijn: het word.", 0],
    ["In hardware is dit beter op te lossen met de juiste configuratie: hardware is niet gebonden aan hele word-bewerkingen.", 0],
    ["Rijndael werkt daarentegen met 32 bit-words zodat een processor die opereert op 32 bit-words simpelweg een word kan lezen en op de juiste plaats wegschrijven.", 0],
    ["In februari 2003 bracht Blue samen met Elton John diens oude hit Sorry seems to be the hardest word uit 1976 uit.", 0],
    ["Als ik boos word, ben ik net zo ridicuul als hij.", 0],
    ["Tot slot de grijsaard? Als ik er een word, is er nog tijd genoeg om daarover te spreken.", 0],
    ["In 2008 heeft hij samen met Pater Moeskroen een liedje gemaakt, getiteld: Het zit er niet in dat ik oud word.", 0],
    ["In \"Hoe word ik succesvoller dan mijn collega's?\" proberen ze antwoord te geven op de vraag welke factoren van invloed zijn op een succesvolle loopbaan.", 0],
    ["Bekend is zijn stelling dat als ik bedrogen word, dan besta ik waarmee hij de beroemde stelling van René Descartes, cogito ergo sum, 1200 jaar voor was.", 0],
    ["Word maakte een muzikale mix van hiphop en funk waarvoor Piet teksten schreef en rapte.", 0],
    ["Word was betrekkelijk succesvol en ondersteunde bands zoals Consolidated, Augurk Players, The Goats en Soul Coughing.", 0],
    ["Zijn grafschrift moest volgens hem luiden: Eindelijk word ik niet meer dommer.", 0],
    ["Ook word je dan niet zelf meer aangemoedigd om je eigen fantasie te gebruiken.", 0],
    ["De beelden met uitspraken als ‘at your service’, en ‘ik word minister-president’, maar ook het televisiedebat met Jeroen Pauw.", 0],
    ["Daar gold, zeker voor de Franse Revolutie: als je voor een dubbeltje geboren bent word je nooit een kwartje.", 0],
    ["Ik word altijd gevraagd voor films - misschien zien ze me op het podium, dat ik zo emotioneel word steeds.", 0],
    ["Hier kwam verandering aan toen ze in 2011 de rol van de rode priesteres Melisandre kreeg aangeboden in het tweede seizoen van de HBO-televisieserie Game of Thrones: Door mijn rol in zo'n goede serie word ik iets serieuzer genomen.", 0],
    ["Samen schreven zij de boeken \"Waarom zit ik niet in oranje?\" en \"Hoe word ik succesvoller dan mijn collega's?\".", 0],
    ["Hierdoor vang je meer wind en word je verder gedragen.", 0],
    ["Daarom word ik Hermes Trismegistus genoemd, omdat ik de drie delen van de filosofie van de gehele wereld bezit.", 0],
    ["Daarom word je verzocht het artikel buiten de Wikipedia naamruimte neer te zetten, waar de normale artikelen staan.", 0],
    ["Staande op de grond of zittend op een stoel word je door de zwaartekracht van de aarde aangetrokken.", 0],
    ["Voor een ambt word je dus uitgekozen of geroepen en vervolgens binnen het protestantisme aangesteld of in meer hiërarchische kerken gewijd.", 0],
    ["In de meeste groepen word je pas leiding op je achttiende.", 0],
    ["Tevens zongen zij voor deze serie de titelsong, hun sinterklaashit \"Ik word later zwarte piet\".", 0],
    ["Ik heb de indruk dat daar op commons strenger tegen opgetreden word.", 0],
    ["In de experimentele popmuziek gebruikt men vaak de term spoken word, waarbij gedichten of verhalen voorgelezen worden op muziek.", 0],
    ["In het tweede geval word je verzocht een bronvermelding aan te geven, in het derde geval is dat verplicht.", 0],
    ["Sommige melodieën zorgen ervoor dat link geteleporteerd word.", 0],
    ["Een rijke Romein, Vettius Agorius Praetextatus, grapte tegen hem: Maak me bisschop van Rome en ik word christen, als reactie op zijn gedrag.", 0],
    ["Aan alle kanten word je daarvoor door cameramensen belaagd.", 0],
    ["Ze gaven daarnaast onder meer een eigen Vrekkenkrant uit en boeken met titels als \"Lekker zuinig\" en \"Hoe word ik een echte vrek?\" over een sobere levenswijze.", 0],
    ["Word je aangevallen, kun je de alarmbel luiden zodat alle dorpelingen zich verschuilen in het stadhuis, kasteel of toren; zo zijn ze niet alleen beschermd, maar kunnen ze ook pijlen schieten op de vijand.", 0],
    ["Ruim drie weken word ik buitengesloten van een project waaraan ik veel werk heb verricht.", 0],
    ["Er duiken grappenmakerop die een naam hanteren die op die van mij lijkt en waarvan men voetstoots aanneemt dat ik het ben en ik word voor nog langere tijd geblokkeerd.", 0],
    ["Dat lijkt me heel goed gezien, en ik zal daaraan verbinden dat ik geen stemcoördinator oid word.", 0],
    ["Markovic schreef enkele dagen voor zijn dood aan zijn broer Aleksandar: Als ik word vermoord, dan is dat 100% zeker de fout van Alain Delon en van zijn peetvader François Marcantoni.", 0],
    ["De Engelse uitspraak van het woord lijkt sterk op de uitspraak van abbey word.", 0],
    ["In februari 2010 baarde Expreszo opzien met de spijbelprotestactie Daar word ik nou ziek van die gericht was tegen de zogenaamde 'enkele-feitconstructie'.", 0],
    ["In Marine Mania word je ondergedompeld in de onderwaterwereld.", 0],

    # With spelling mistake
    # Incorrect inversions 'word' => 'wordt'
    ["Daar wordt ik hartstikke gek van.", 1],
    ["Ik hoop dat dit dus niet bij jou gedaan wordt.", 1],
    ["Zo verving von Baader Descartes' cogito ergo sum door cogitor ergo sum: ik wordt gedacht, dus ik ben.", 1],
    ["Ik ga door tot Parijs en zal winnen, tenzij ik vermoord wordt.", 1],
    ["Hij zou op de dag van de verkiezing tegen zijn moeder gezegd hebben Of ik keer terug als pontiflex, of ik wordt voor altijd verbannen.", 1],
    ["Rond het thema van de Vlaamse onafhankelijkheid schreef hij ‘Operatie Vlaamse Onafhankelijkheid‘, een verslag van de evaluatiedag ‘Volk, wordt staat’ dat op 24 november 2007 plaatsvond in het Federale Parlement.", 1],
    ["Als je goed leeft, kun je na de dood in een hogere kaste terecht komen en uiteindelijk wordt je bevrijd van het aardse leven om samen te leven met god.", 1],
    ["Vertaling: Ik wordt beschermd of hij beschermt mij.", 1],
    ["Als je van deze appels eet, wordt je onsterfelijk.", 1],
    ["In 2008 begon BNN met een grootse campagne onder het motto 'Koop dit, wordt lid'.", 1],
    ["Nu pas wordt ik me ervan bewust dat mijn werk in zwart, wit en kleine kleurvlakken alleen maar 'tekenen' in olieverf is geweest.", 1],
    ["Wel liet hij ooit in een interview optekenen: Oud met een d wordt ik niet waarschijnlijk, gezien de grote hoeveelheid koffie en sigaretten die ik consumeer, en de geringe hoeveelheid slaap die ik geniet.", 1],
    ["In 2008 verscheen het eerste luisterboek van doctorandus P, waarop hij zelf drie eigen verhalen voorleest: Sven de Bevrijder, Ik wordt vermoord en Tiens, Tiens, afgewisseld met een aantal ollekebollekes.", 1],
    ["Juridisch wordt je bij een koop dus nog geen eigenaar!", 1],
    ["Zuiderspel wordt georganiseerd in Hotel and Congrescentrum Koningshof te Veldhoven.", 1],
    ["Nescio's bevlogen proza wordt in de tweede alinea komisch-ruw onderbroken door Bavink: Toen zei Bavink: 'Ik wordt een beroemd man,' zooals een ander zou zeggen: 'Ze hebben me een dubbeltje te veel afgezet,' en we voelden ons bekocht, alle drie, Bavink, Bekker en ik.", 1],
    ["In het West-Vlaams zegt men dus niet ik wordt ziek, maar men zegt ik kom ziek.", 1],
    ["Vroeg of laat wordt ik natuurlijk vergeten, zei Havel bij die gelegenheid, maar gelukkig is er nog altijd die foto waarop ik met Arnold Schwarzenegger sta.", 1],
    ["Want 'stel je je kwetsbaar op, dan wordt je gekwetst', aldus Cruijff.", 1],
    ["Daar wordt je met al je idealen natuurlijk ontzettend agressief van, maar er is nu eenmaal geen vooropleiding voor wethouder.", 1],
    ["Na de uitspraak \"Ik wordt daar zo moe van\" valt hij meestal direct in slaap.", 1],
    ["Dachau heeft een blijvend stempel op hem gedrukt: Banger wordt ik voor mijn eigen wezen, Dachau schoof een raster voor mijn ziel en wie daarin opgenomen is geweest, zal de dood tot zijn dood met zich meedragen.", 1],
    ["Op zijn sterfbed sprak hij de woorden uit: Aan mijn geleerdheid heb ik nu niets meer; mijn dogmatiek baat mij ook nu niet meer; alleen door het geloof wordt ik zalig.", 1],
    ["Römer vond dat mensen daar behandeld wordten als gebruiksvoorwerpen: Eerst word je in de watten gelegd en later laten ze je net zo hard weer vallen.", 1],
    ["Ik wordt nog meer gehaat dan voorheen, omdat ik nu goed bij kas ben.", 1],
    ["Onderdelen van de set zoals de kippen en bloemen werden meegegeven aan het publiek en verloot onder mensen die lid waren gewordten van BNN tijdens de “Koop dit, word lid”-actie.", 1],
    ["Ik wordt misschien nog eens medeplichtig gehouden, als Mitja zich aan uw vader mocht vergrijpen.", 1],
    ["Om de nieuwe SIMD-instructies te kunnen implementeren heeft Intel 4 nieuwe datatypen ingevoerd, de packed versies van de byte, wordt, doubleword en quadword.", 1],
    ["Natuurlijk ben je lid van een familie door geboorte en wordt je op een bepaalde leeftijd ingewijd, door middel van rituelen en een eed voor het leven, maar je kunt ook bij een familie komen als je door de familie wordt geaccepteerd en ritueel wordt omgedoopt.", 1],
    ["Houd er ook rekening mee dat de spellingcontrole wel werkwoordsvormen die echt fout zijn markeert, zoals heeft gebruikd, maar niet de werkwoordsvormen waarin wel goede woorden staan maar die grammaticaal niet correct zijn: het wordt.", 1],
    ["In hardware is dit beter op te lossen met de juiste configuratie: hardware is niet gebonden aan hele wordt-bewerkingen.", 1],
    ["Rijndael werkt daarentegen met 32 bit-wordts zodat een processor die opereert op 32 bit-words simpelweg een word kan lezen en op de juiste plaats wegschrijven.", 1],
    ["In februari 2003 bracht Blue samen met Elton John diens oude hit Sorry seems to be the hardest wordt uit 1976 uit.", 1],
    ["Als ik boos wordt, ben ik net zo ridicuul als hij.", 1],
    ["Tot slot de grijsaard? Als ik er een wordt, is er nog tijd genoeg om daarover te spreken.", 1],
    ["In 2008 heeft hij samen met Pater Moeskroen een liedje gemaakt, getiteld: Het zit er niet in dat ik oud wordt.", 1],
    ["In \"Hoe wordt ik succesvoller dan mijn collega's?\" proberen ze antwoord te geven op de vraag welke factoren van invloed zijn op een succesvolle loopbaan.", 1],
    ["Bekend is zijn stelling dat als ik bedrogen wordt, dan besta ik waarmee hij de beroemde stelling van René Descartes, cogito ergo sum, 1200 jaar voor was.", 1],
    ["Wordt maakte een muzikale mix van hiphop en funk waarvoor Piet teksten schreef en rapte.", 1],
    ["Wordt was betrekkelijk succesvol en ondersteunde bands zoals Consolidated, Augurk Players, The Goats en Soul Coughing.", 1],
    ["Zijn grafschrift moest volgens hem luiden: Eindelijk wordt ik niet meer dommer.", 1],
    ["Ook wordt je dan niet zelf meer aangemoedigd om je eigen fantasie te gebruiken.", 1],
    ["De beelden met uitspraken als ‘at your service’, en ‘ik wordt minister-president’, maar ook het televisiedebat met Jeroen Pauw.", 1],
    ["Daar gold, zeker voor de Franse Revolutie: als je voor een dubbeltje geboren bent wordt je nooit een kwartje.", 1],
    ["Ik wordt altijd gevraagd voor films - misschien zien ze me op het podium, dat ik zo emotioneel word steeds.", 1],
    ["Hier kwam verandering aan toen ze in 2011 de rol van de rode priesteres Melisandre kreeg aangeboden in het tweede seizoen van de HBO-televisieserie Game of Thrones: Door mijn rol in zo'n goede serie wordt ik iets serieuzer genomen.", 1],
    ["Samen schreven zij de boeken \"Waarom zit ik niet in oranje?\" en \"Hoe wordt ik succesvoller dan mijn collega's?\".", 1],
    ["Hierdoor vang je meer wind en wordt je verder gedragen.", 1],
    ["Daarom wordt ik Hermes Trismegistus genoemd, omdat ik de drie delen van de filosofie van de gehele wereld bezit.", 1],
    ["Daarom wordt je verzocht het artikel buiten de Wikipedia naamruimte neer te zetten, waar de normale artikelen staan.", 1],
    ["Staande op de grond of zittend op een stoel wordt je door de zwaartekracht van de aarde aangetrokken.", 1],
    ["Voor een ambt wordt je dus uitgekozen of geroepen en vervolgens binnen het protestantisme aangesteld of in meer hiërarchische kerken gewijd.", 1],
    ["In de meeste groepen wordt je pas leiding op je achttiende.", 1],
    ["Tevens zongen zij voor deze serie de titelsong, hun sinterklaashit \"Ik wordt later zwarte piet\".", 1],
    ["Ik heb de indruk dat daar op commons strenger tegen opgetreden wordt.", 1],
    ["In de experimentele popmuziek gebruikt men vaak de term spoken wordt, waarbij gedichten of verhalen voorgelezen worden op muziek.", 1],
    ["In het tweede geval wordt je verzocht een bronvermelding aan te geven, in het derde geval is dat verplicht.", 1],
    ["Sommige melodieën zorgen ervoor dat link geteleporteerd wordt.", 1],
    ["Een rijke Romein, Vettius Agorius Praetextatus, grapte tegen hem: Maak me bisschop van Rome en ik wordt christen, als reactie op zijn gedrag.", 1],
    ["Aan alle kanten wordt je daarvoor door cameramensen belaagd.", 1],
    ["Ze gaven daarnaast onder meer een eigen Vrekkenkrant uit en boeken met titels als \"Lekker zuinig\" en \"Hoe wordt ik een echte vrek?\" over een sobere levenswijze.", 1],
    ["Wordt je aangevallen, kun je de alarmbel luiden zodat alle dorpelingen zich verschuilen in het stadhuis, kasteel of toren; zo zijn ze niet alleen beschermd, maar kunnen ze ook pijlen schieten op de vijand.", 1],
    ["Ruim drie weken wordt ik buitengesloten van een project waaraan ik veel werk heb verricht.", 1],
    ["Er duiken grappenmakerop die een naam hanteren die op die van mij lijkt en waarvan men voetstoots aanneemt dat ik het ben en ik wordt voor nog langere tijd geblokkeerd.", 1],
    ["Dat lijkt me heel goed gezien, en ik zal daaraan verbinden dat ik geen stemcoördinator oid wordt.", 1],
    ["Markovic schreef enkele dagen voor zijn dood aan zijn broer Aleksandar: Als ik wordt vermoord, dan is dat 100% zeker de fout van Alain Delon en van zijn peetvader François Marcantoni.", 1],
    ["De Engelse uitspraak van het woord lijkt sterk op de uitspraak van abbey wordt.", 1],
    ["In februari 2010 baarde Expreszo opzien met de spijbelprotestactie Daar wordt ik nou ziek van die gericht was tegen de zogenaamde 'enkele-feitconstructie'.", 1],
    ["In Marine Mania wordt je ondergedompeld in de onderwaterwereld.", 1],

    # vinden and lopen
    # Correct spelling
    ["Ik vind dit toch niet zo super.", 0],
    ["Wat vind jij van al zijn geleuter?", 0],
    ["Hoe vind ik best de volgende stap?", 0],
    ["Het is toch best erg dat hij dat nu vindt.", 0],
    ["Hoe zwaar vindt hij die tegenslag?", 0],

    ["Ik loop er niet zo snel voorbij.", 0],
    ["Loop jij ook zo rustig?", 0],
    ["Je loopt daar makkelijk tegenaan.", 0],
    ["Jij loopt weer helemaal naar achter?", 0],
    ["En daarom loopt hij er niet recht naartoe.", 0],

    # With spelling mistake
    ["Ik vindt dit toch niet zo super.", 1],
    ["Wat vindt jij van al zijn geleuter?", 1],
    ["Hoe vindt ik best de volgende stap?", 1],
    ["Het is toch best erg dat hij dat nu vind.", 0],
    ["Hoe zwaar vind hij die tegenslag?", 1],

    ["Ik loopt er niet zo snel voorbij.", 1],
    ["Loopt jij ook zo rustig?", 1],
    ["Je loop daar makkelijk tegenaan.", 1],
    ["Jij loop weer helemaal naar achter?", 1],
    ["En daarom loop hij er niet recht naartoe.", 1],

    # Correct spelling
    ["Ik vind dit wel leuk, eigenlijk.", 0],
    ["Wat vind jij van al dat gedoe?", 0],
    ["Hoe vind ik he snelste de uitgang?", 0],
    ["Het is toch heftig dat jij dit hier vindt.", 0],
    ["Hoe zwaar vindt hij die tegenslag?", 0],

    ["Ik loop er niet zo snel voorbij.", 0],
    ["Loop jij even naar de bakker?", 0],
    ["Je loopt nu een pak vlotter dan vroeger.", 0],
    ["Jij loopt toch niet echt met die schoenen?", 0],
    ["En waarom loopt hij er nu niet direct heen?", 0],

    # With spelling mistake
    ["Ik vind dit wel leuk, eigenlijk.", 1],
    ["Wat vindt jij van al dat gedoe?", 1],
    ["Hoe vindt ik he snelste de uitgang?", 1],
    ["Het is toch heftig dat jij dit hier vind.", 1],
    ["Hoe zwaar vind hij die tegenslag?", 1],

    ["Ik loopt er niet zo snel voorbij.", 1],
    ["Loopt jij even naar de bakker?", 1],
    ["Je loop nu een pak vlotter dan vroeger.", 1],
    ["Jij loop toch niet echt met die schoenen?", 1],
    ["En waarom loop hij er nu niet direct heen?", 1],

    # More sentences from nl.wikipedia
    # Correct spelling
    ["Deze vorm van veehouderij wordt nog steeds beoefend in delen van Beieren, Oostenrijk, Slovenië, Italië en Zwitserland.", 0],
    ["De Sloveense alpinistenbond onderhoudt ruim 150 berghutten.", 0],
    ["De autosnelweg R1 vormt de Ring rond Antwerpen en verbindt de A1/E19, de A12, de A21/E34, de A13/E313, de A1/E19, de A12 , e A14/E17 en de E34 met elkaar.", 0],
    ["De VOC hoopte dat door deze reis dit onbekende continent voor de handel geopend en vervolgens geëxploiteerd zou kunnen worden.", 0],
    ["Dit monument werd na een opknapbeurt in 1992 door koningin Beatrix tijdens een staatsbezoek aan Nieuw-Zeeland opnieuw onthuld.", 0],
    ["Een dag later meldde uitgeverij De Arbeiderspers dat dit niet klopte, en dat een citaat uit zijn verband getrokken was.", 0],
    ["Een acteur is iemand die een personage uitbeeldt in een verhaal of rollenspel.", 0],
    ["Iemand als Sylvester Stallone wordt vaak gecast in stoere rollen, terwijl Dustin Hoffman meestal comedy's speelt.", 0],
    ["Over wat acteren precies is en hoe het beste resultaat wordt bereikt, wordt verschillend gedacht.", 0],
    ["De commissie maakte op 2 juni 2008 haar rapport bekend.", 0],
    ["Houd hier dus geen pleidooien voor een bepaald standpunt, maar verwijs beknopt naar een overlegpagina.", 0],
    ["Stapelverhaaltjes beginnen en eindigen met hetzelfde zinnetje en variëren daarop zoals in Geert De Kockeres 'Houd de dief'.", 0],
    ["Suggestie: houd deze pagina in een apart venster bij de hand tijdens het schrijven van teksten.", 0],
    ["Bouw geen torenhoge flat van twintig verdiepingen, maar houd de structuur zo plat en overzichtelijk mogelijk.", 0],
    ["Als je een hekel hebt aan regels en wat daarop lijkt, houd dan hier op met lezen, en ga je eigen, vrije gang.", 0],
    ["Stel je daarbij altijd de vraag wat jouw website voor meerwaarde heeft voor een link in een encyclopedisch artikel en houd je aan de hierboven omschreven algemene richtlijnen.", 0],

    # With spelling mistake
    ["Deze vorm van veehouderij word nog steeds beoefend in delen van Beieren, Oostenrijk, Slovenië, Italië en Zwitserland.", 1],
    ["De Sloveense alpinistenbond onderhoud ruim 150 berghutten.", 1],
    ["De autosnelweg R1 vormt de Ring rond Antwerpen en verbind de A1/E19, de A12, de A21/E34, de A13/E313, de A1/E19, de A12 , e A14/E17 en de E34 met elkaar.", 1],
    ["De VOC hoopte dat door deze reis dit onbekende continent voor de handel geopent en vervolgens geëxploiteerd zou kunnen worden.", 1],
    ["Dit monument werd na een opknapbeurt in 1992 door koningin Beatrix tijdens een staatsbezoek aan Nieuw-Zeeland opnieuw onthult.", 1],
    ["Een dag later melde uitgeverij De Arbeiderspers dat dit niet klopte, en dat een citaat uit zijn verband getrokken was.", 1],
    ["Een acteur is iemand die een personage uitbeeld in een verhaal of rollenspel.", 1],
    ["Iemand als Sylvester Stallone word vaak gecast in stoere rollen, terwijl Dustin Hoffman meestal comedy's speelt.", 1],
    ["Over wat acteren precies is en hoe het beste resultaat word bereikt, wordt verschillend gedacht.", 1],
    ["De commissie maakte op 2 juni 2008 haar rapport bekent.", 1],
    ["Houdt hier dus geen pleidooien voor een bepaald standpunt, maar verwijs beknopt naar een overlegpagina.", 1],
    ["Stapelverhaaltjes beginnen en eindigen met hetzelfde zinnetje en variëren daarop zoals in Geert De Kockeres 'Houdt de dief'.", 1],
    ["Suggestie: houdt deze pagina in een apart venster bij de hand tijdens het schrijven van teksten.", 1],
    ["Bouw geen torenhoge flat van twintig verdiepingen, maar houdt de structuur zo plat en overzichtelijk mogelijk.", 1],
    ["Als je een hekel hebt aan regels en wat daarop lijkt, houdt dan hier op met lezen, en ga je eigen, vrije gang.", 1],
    ["Stel je daarbij altijd de vraag wat jouw website voor meerwaarde heeft voor een link in een encyclopedisch artikel en houdt je aan de hierboven omschreven algemene richtlijnen.", 1],
]

train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Optional model configuration
# with 20 epochs, takes 60 minutes on laptop CPU
model_args = {
    "num_train_epochs": 20,
    "overwrite_output_dir": 0, # we are moving the dir to a named dir now
    "output_dir": "outputs/RoBERTa"
}

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "pdelobelle/robBERT-base", args=model_args, use_cuda=False,
)
print(type(model))
# <class 'simpletransformers.classification.classification_model.ClassificationModel'>

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(model_outputs) # see above

# Print the incorrect predictions
for wrong_prediction in wrong_predictions:
    print(wrong_prediction.text_a)
    print(wrong_prediction.label)
