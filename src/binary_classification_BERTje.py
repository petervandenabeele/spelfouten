from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
# Correct Dutch sentences have a '0' label
# Sentences with a "dt" mistake for worden (word vs. wordt) have a '1' label
train_data = [
    ["Ik word is zonder t", 0],
    ["Ik word warm.", 0],
    ["Ik word enthousiast.", 0],
    ["Ik word eigenaar van een nieuwe kat.", 0],
    ["Hierdoor word ik ook helemaal verrast.", 0],
    ["Hoe word ik eigenlijk geholpen?", 0],
    ["Waarom word ik van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder word ik hierdoor afgezet?", 0],

    ["Ik wordt is zonder t", 1],
    ["Ik wordt warm.", 1],
    ["Ik wordt enthousiast.", 1],
    ["Ik wordt eigenaar van een nieuwe kat.", 1],
    ["Hierdoor wordt ik ook helemaal verrast.", 1],
    ["Hoe wordt ik eigenlijk geholpen?", 1],
    ["Waarom wordt ik van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder wordt ik hierdoor afgezet?", 1],

    ["Jij wordt is met t", 0],
    ["Jij wordt warm.", 0],
    ["Jij wordt enthousiast.", 0],
    ["Jij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor word jij ook helemaal verrast.", 0],
    ["Hoe word jij eigenlijk geholpen?", 0],
    ["Waarom word jij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder word jij hierdoor afgezet?", 0],

    ["Jij word is met t", 1],
    ["Jij word warm.", 1],
    ["Jij word enthousiast.", 1],
    ["Jij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor wordt jij ook helemaal verrast.", 1],
    ["Hoe wordt jij eigenlijk geholpen?", 1],
    ["Waarom wordt jij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder wordt jij hierdoor afgezet?", 1],

    ["Hij wordt is met t", 0],
    ["Hij wordt warm.", 0],
    ["Hij wordt enthousiast.", 0],
    ["Hij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor wordt hij ook helemaal verrast.", 0],
    ["Hoe wordt hij eigenlijk geholpen?", 0],
    ["Waarom wordt hij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder wordt hij hierdoor afgezet?", 0],

    ["Hij word is met t", 1],
    ["Hij word warm.", 1],
    ["Hij word enthousiast.", 1],
    ["Hij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor word hij ook helemaal verrast.", 1],
    ["Hoe word hij eigenlijk geholpen?", 1],
    ["Waarom word hij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder word hij hierdoor afgezet?", 1],

    ["Zij wordt is met t", 0],
    ["Zij wordt warm.", 0],
    ["Zij wordt enthousiast.", 0],
    ["Zij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor wordt zij ook helemaal verrast.", 0],
    ["Hoe wordt zij eigenlijk geholpen?", 0],
    ["Waarom wordt zij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder wordt zij hierdoor afgezet?", 0],

    ["Zij word is met t", 1],
    ["Zij word warm.", 1],
    ["Zij word enthousiast.", 1],
    ["Zij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor word zij ook helemaal verrast.", 1],
    ["Hoe word zij eigenlijk geholpen?", 1],
    ["Waarom word zij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder word zij hierdoor afgezet?", 1],

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

    # Incorrect inversions
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
    ["Nescio's bevlogen proza wordtt in de tweede alinea komisch-ruw onderbroken door Bavink: Toen zei Bavink: 'Ik word een beroemd man,' zooals een ander zou zeggen: 'Ze hebben me een dubbeltje te veel afgezet,' en we voelden ons bekocht, alle drie, Bavink, Bekker en ik.", 1],
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
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data

## RESULTS on this version with approx. 200 training examples (64 synthetic, 140 from nl.wikipedia)
## and approx. 46 validation examples (16 synthetic, 30 (independent) ones from nl.wikipedia)

## Validation accuracy of 45/46 => 97.8% accuracy (?)
# {'mcc': 0.957427107756338, 'tp': 23, 'tn': 22, 'fp': 1, 'fn': 0, 'eval_loss': 0.02614415737237626}
# [[ 4.2059097  -3.7627044 ]
#  [-3.5665493   3.1585    ]
#  [ 4.212945   -3.6560168 ]
#  [-3.6388679   3.3253736 ]
#  [ 2.7759726  -2.4389825 ]
#  [-3.2775264   2.94584   ]
#  [ 4.0644474  -3.6081064 ]
#  [-3.0999212   2.9003782 ]
#  [ 3.7575562  -3.2989907 ]
#  [-3.3962812   2.9213512 ]
#  [ 1.8863125  -1.5268269 ]
#  [-3.2213383   2.9490423 ]
#  [ 3.5826302  -3.0704281 ]
#  [-3.3908744   2.975732  ]
#  [-0.34237236  0.43810555]  *INCORRECT* => "Wordt zij volgend jaar ook uitgenodigd?""
#  [-3.3102717   3.0289602 ]
#  [ 4.2628994  -3.6565843 ]
#  [ 4.3029656  -3.6985798 ]
#  [ 3.8944387  -3.5548325 ]
#  [ 4.2086067  -3.709293  ]
#  [ 4.234384   -3.6440425 ]
#  [ 4.1677513  -3.7484157 ]
#  [ 4.0783415  -3.624547  ]
#  [ 4.1836805  -3.5575871 ]
#  [ 4.198094   -3.6942775 ]
#  [ 4.0628657  -3.6311564 ]
#  [ 4.2124763  -3.5904891 ]
#  [ 4.214889   -3.641985  ]
#  [ 4.250997   -3.698926  ]
#  [ 4.2112365  -3.7224946 ]
#  [ 4.231102   -3.6591349 ]
#  [-3.111953    2.6416206 ]
#  [-2.9442997   2.6951098 ]
#  [-3.093072    2.7891457 ]
#  [-3.3627703   3.0261097 ]
#  [-3.6477983   3.2423062 ]
#  [-3.5561976   3.1162627 ]
#  [-3.4674215   3.2619922 ]
#  [-2.602563    2.5227032 ]
#  [-3.4936633   3.1744359 ]
#  [-3.0891414   2.7304373 ]
#  [-3.4588      3.1527596 ]
#  [-3.1135874   2.6390429 ]
#  [-3.6027222   3.1697285 ]
#  [-3.5643284   3.3636334 ]
#  [-3.5316904   3.2791054 ]]

# This is the single wrong prediction in the evaluation set
# Wordt zij volgend jaar ook uitgenodigd?
# 0


eval_data = [
    ["Ik word volgend jaar ook getest.", 0],          #  [ 0.934536   -0.79441744]
    ["Ik wordt helemaal naar hier gehaald.", 1],      #  [-0.38130385  0.34947482]
    ["Word ik volgend jaar ook uitgenodigd?", 0],     #  [ 0.56628114 -0.31090865]
    ["Wordt ik nu al opgeroepen?", 1], # *INCORRECT*  #  [ 0.24857064 -0.13078722]

    ["Jij wordt volgend jaar ook getest.", 0],        #  [ 1.1392815  -0.768657  ]
    ["Jij word helemaal naar hier gehaald.", 1],      #  [-1.1988978   1.0266285 ]
    ["Word jij volgend jaar ook uitgenodigd?", 0],    #  [ 0.38991755 -0.34029955]
    ["Wordt jij nu al opgeroepen?", 1], # *INCORRECT* #  [ 0.34916613 -0.29456055]

    ["Hij wordt volgend jaar ook getest.", 0],        #  [ 1.3438416  -0.99393845]
    ["Hij word helemaal naar hier gehaald.", 1],      #  [-1.3128942   1.3501518 ]
    ["Wordt hij volgend jaar ook uitgenodigd?", 0],   #  [ 1.1691108  -0.7768973 ]
    ["Word hij nu al opgeroepen?", 1],                #  [-0.2947425   0.29919532]

    ["Zij wordt volgend jaar ook getest.", 0],        #  [ 1.4490408  -1.1109018 ]
    ["Zij word helemaal naar hier gehaald.", 1],      #  [-1.4008088   1.3720468 ]
    ["Wordt zij volgend jaar ook uitgenodigd?", 0],   #  [ 1.3757272  -1.0468719 ]
    ["Word zij nu al opgeroepen?", 1],                #  [-0.54108465  0.3590323 ]

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
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
# with 10 epochs, took a few minutes to train on laptop CPU
model_args = {
    "num_train_epochs": 10,
    "overwrite_output_dir": 1,
}

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-dutch-cased", args=model_args, use_cuda=False,
)
print(type(model))
# <class 'simpletransformers.classification.classification_model.ClassificationModel'>

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(model_outputs) # see above
# =>  {'mcc': 0.957427107756338, 'tp': 23, 'tn': 22, 'fp': 1, 'fn': 0, 'eval_loss': 0.02614415737237626}

# See above, the single wrong prediction for "Wordt zij volgend jaar ook uitgenodigd?"
for wrong_prediction in wrong_predictions:
    print(wrong_prediction.text_a)
    print(wrong_prediction.label)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Ik wordt nieuwsgierig."])
print(predictions)
print(raw_outputs)
# [1]
# [[-3.4197278  3.2459447]] => strongly correct

predictions, raw_outputs = model.predict(["Wordt jij ook enthousiast?"])
print(predictions)
print(raw_outputs)
# [1]
# [[-2.9257555  2.6328616]] => strongly correct

predictions, raw_outputs = model.predict(["Wordt zij hierdoor bekend?"])
print(predictions)
print(raw_outputs)
# [1]
# [[-0.18000953  0.8206537 ]] => *WEAKLY INCORRECT* ; the "wordt zij ..." is clearly failing for this model

predictions, raw_outputs = model.predict(["Ik word enthousiast."])
print(predictions)
print(raw_outputs)
# [0]
# [[ 4.1996617 -3.7211807]] => strongly correct