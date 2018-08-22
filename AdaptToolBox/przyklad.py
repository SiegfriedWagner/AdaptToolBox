import adaptation.UML
alpha = adaptation.UML.UMLParameter(min_value=-20,
                             max_value=20,
                             n=61,
                             scale='lin',
                             dist='flat',
                             mu=0,
                             std=20)
beta = adaptation.UML.UMLParameter(min_value=1,
                    max_value=20,
                    n=41,
                    scale='log',
                    dist='norm',
                    mu=0.5,
                    std=2)
gamma = adaptation.UML.UMLParameter(value = 0.5)
lamb = adaptation.UML.UMLParameter(scale="lin",
                    dist="flat",
                    min_value=0,
                    max_value=0.1,
                    mu=0,
                    std=0.1,
                    n=5)
uml = adaptation.UML.GaussianUML(safemode=False,
                                  max_stimuli=30,
                                  min_stimuli=-30,
                                  value=25,
                                  method='mean',
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  lamb=lamb)
uml.update(1)
