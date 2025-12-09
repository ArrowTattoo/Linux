/* 
    branch & Bound pour le SPP (base generalisable a un autre pb en variables 01)
    ODC 2023-2024
    Xavier Gandibleux
*/

#include "gurobi_c++.h"
#include <queue>
#include <list>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

// une instance
struct tInstance {
    string  fname;
    int I;
    int J;
    double *  c;	
    double ** a;
    double *  b;
} ;

// une solution
struct tSolution {
    double   z;    // valeur de la solution
    double * x;    // variables de la solution
} ;

// un sommet du branch-and-bound
struct tNoeud {
    int        id;           // numero identifiant
    tSolution  bDuale;       // borne duale
    GRBModel * model;        // modele correspondant au noeud 
};

// prototypes -----------------------------------------------------------------
tInstance chargeInstance(int numero);
tInstance SPPparser(string fname);
void SPPafficheInstance(tInstance instance);

GRBModel setupModel(GRBEnv& env, tInstance instance);
void displayResults(GRBModel& model);
void displayResults2(GRBModel * model);
int heuristiqueSeparation(GRBModel& model, double tol);
void infererNoeud(GRBModel& model, char DroiteGauche, int jPos, double tol, tSolution * bestSolution, list<tNoeud> * noeudsActifs, int * cptNoeuds);

int estApprox0(double x, double tol);
int estApprox1(double x, double tol);
//int estSolution01(double * x, int J, double tol);
int estSolution01(GRBModel&  m, double tol);
double * clone_X(GRBModel&  m);

void SPPlibereMemoire(tInstance instance); 

int main(int argc, char *argv[])
{
    // 0. Initialisation ------------------------------------------------------

    // meilleure solution trouvee
    tSolution bestSolution;

    // tolerance sur le test de l'egalite d'une valeur flottante
    double tol = 1e-06;

    // liste de noeuds actifs
    list<tNoeud> noeudsActifs;

    // compteur de noeuds construits
    int cptNoeuds = 0;

    // cree un environement 
    GRBEnv env = GRBEnv(true);
    //env.set("LogFile", "B&B.log");
    env.start();

    // charge une instance de SPP 
    tInstance instance = chargeInstance(3); // directement

    string fname = "pb_100rnd0100.dat";
    //tInstance instance  = SPPparser(fname); // sur fichier
    
    SPPafficheInstance(instance);

    // initialise un modele Gurobi de SPP 
    GRBModel model = setupModel(env, instance);

    // pas d'heuristique primale => pas de valeur pour la borne primale pour demarrer
    bestSolution.z = -1.0; 
    bestSolution.x = NULL;


    // 1. Traitement de la racine ---------------------------------------------

    cout << "\nR A C I N E ------------------------------------ \n" << endl;

    // optimise le modele 
    model.optimize();

    // affiche le resultat de l'optimisation 
    displayResults(model);
    
    // initialise l'arborescence
    tNoeud racine;
    racine.id         = cptNoeuds;                        //  numero identifiant du noeud
    racine.bDuale.z   = model.get(GRB_DoubleAttr_ObjVal); //  z pour relaxation lineaire
    racine.bDuale.x   = clone_X(model);                   //  x pour relaxation lineaire
    cptNoeuds         = cptNoeuds+1;

    // verifie l'integralite de la solution a la racine
    if ( estSolution01(model, tol)==1 )
    {
        // solution admissible entiere trouvee
        cout << "Solution entiere trouvee a la racine => noeud sonde par optimalite => fin" << endl;
        bestSolution.z = model.get(GRB_DoubleAttr_ObjVal);
        bestSolution.x = clone_X(model);
    }
    else
    {
        // separer : choisir la variable fractionnaire la plus proche d'un entier
        int jPos = heuristiqueSeparation(model, tol);
        cout << "Separation sur var x[" << jPos << "]" << endl; 

        GRBVar * x = model.getVars() ;
        char DroiteGauche;

        // noeud droite -------------------------------------------------------
        DroiteGauche = 'D';
        infererNoeud(model, DroiteGauche, jPos, tol, & bestSolution, & noeudsActifs, & cptNoeuds);
        cout << "\nNbre de noeuds actifs : "<< noeudsActifs.size()<<endl;

        // noeud gauche -------------------------------------------------------
        DroiteGauche = 'G';
        infererNoeud(model, DroiteGauche, jPos, tol, & bestSolution, & noeudsActifs, & cptNoeuds);
        cout << "\nNbre de noeuds actifs : "<< noeudsActifs.size()<<endl;
    }


    // 2. Traitement des noeuds actifs ----------------------------------------
    while (noeudsActifs.size()>0)
    {
        cout << "\nNbre de noeuds actifs : "<< noeudsActifs.size()<<endl;
        cout << "Info des noeuds actifs :" << endl;
        for(list<tNoeud>::iterator it = noeudsActifs.begin(); it != noeudsActifs.end(); it++)
            {
                cout << "  id  " << it->id 
                     << " | z  " << it->bDuale.z << endl;
	        }

        // liste noeuds actifs non vide => recherche du noeud le plus prometteur (best-node)
        cout << "\nSelection d'un noeud actif : ----------------- " << endl; 
        list<tNoeud>::iterator itMax;
        double vMax = 0.0; 

        // recherche du best-node ---------------------------------------------
        for(list<tNoeud>::iterator it = noeudsActifs.begin(); it != noeudsActifs.end(); it++)
            if (it->bDuale.z > vMax)
            {
                itMax = it;
                vMax = it->bDuale.z;
            }

        cout << "ID & z du noeud best-node selectionne : " << itMax->id << " " << vMax << endl;
        /*
        cout << "  id " << itMax->id 
             << " | z  " << itMax->bDuale.z 
             << " | st " << itMax->model->get(GRB_IntAttr_Status) << endl; 

        // (re)optimise le best-node (point a ameliorer si possible sauvegarder info de resolution) 
        itMax->model->optimize();

        cout << " st2 " << itMax->model->get(GRB_IntAttr_Status) << endl; 
        cout << "#var " << itMax->model->get(GRB_IntAttr_NumVars) <<  endl;
        cout << " Obj " << itMax->model->get(GRB_DoubleAttr_ObjVal) << endl;
        displayResults2(itMax->model);


        // separer : choisir la variable fractionnaire la plus proche d'un entier
        int jPos = heuristiqueSeparation(*(itMax->model), tol);
        cout << "Separation sur var x[" << jPos << "]" << endl; 

        GRBVar * x = itMax->model->getVars() ;
        */

        //improve

        if (floor(vMax) <= bestSolution.z) {
            cout << " (Pruned by bound)" << endl;
            delete itMax->model; 
            free(itMax->bDuale.x);
            noeudsActifs.erase(itMax);
            continue;
        }
        GRBModel *currentModel = itMax->model;

        char DroiteGauche;

        // noeud droite -------------------------------------------------------
        DroiteGauche = 'D';
        infererNoeud(*(itMax->model), DroiteGauche, jPos, tol, & bestSolution, & noeudsActifs, & cptNoeuds);
        //cout << "\nNbre de noeuds actifs : "<< noeudsActifs.size()<<endl;

        // noeud gauche -------------------------------------------------------
        DroiteGauche = 'G';
        infererNoeud(*(itMax->model), DroiteGauche, jPos, tol, & bestSolution, & noeudsActifs, & cptNoeuds);
        //cout << "\nNbre de noeuds actifs : "<< noeudsActifs.size()<<endl;

        // retire de la liste des noeuds actifs le best-node qui a ete traite
        noeudsActifs.erase (itMax);      
    }
  

    // 3. Affichage de la solution optimale -----------------------------------

    cout << "\nSolution Optimale : ---------------------------- " << endl;

    int nbrVar = instance.J;

    if (bestSolution.z != -1.0)
    {
        cout << "\nValeur" << endl;
        cout << "  Obj= " << bestSolution.z << endl;
        
        cout << "\nVariables" << endl;
        for (int j=0; j<nbrVar; j++)
        {
            cout << "  "
                 << "x[" << j+1 << "]"
                 << " = "  
                 << bestSolution.x[j]
                 << endl;
        }
    }
    else
        cout << "\nPas de solution entiere trouvee => instance sans solution\n" << endl;

    // 4. Liberation de la memoire allouee dynamiquement ----------------------

    SPPlibereMemoire(instance);

    // to do supprimer modele Gurobi
    // GRBVar * vars = model.getVars() ;
    // delete[] vars;    

  return 0;
}


// ============================================================================
// ============================================================================

// ----------------------------------------------------------------------------
// Alloue dynamiquement la memoire pour une instance et affecte les valeurs dans instance

tInstance affecterValeurs(const int I, const int J, double *vc, double *va)
{
    tInstance instance;

    instance.I = I;
    instance.J = J; 

    // allocation dynamique
    instance.c = (double *) malloc( instance.J * sizeof(double) );

    instance.a = (double **) malloc( instance.I * sizeof(double *) );
    for(int i = 0; i < instance.I; i++)
        instance.a[i] = (double *) malloc( instance.J * sizeof(double) );

    instance.b = (double *) malloc( instance.I * sizeof(double) );

    // affectation des valeurs
    for (int j=0; j<J; j++)
        instance.c[j]= vc[j];

    for (int i=0; i<I; i++)
        for (int j=0; j<J; j++)
            instance.a[i][j] = *(va+i*J+j);

    for (int i=0; i<I; i++)   
        instance.b[i] = 1;   

    return instance;    
}


// ----------------------------------------------------------------------------
// charge une instance numerique et initialise la structure instance

tInstance chargeInstance(int numero)
{
    tInstance instance;

    if ( numero == 1 )
    {
        const int I1 = 3;
        const int J1 = 3;  
        double vc1[]    = { 1,  1,  1};	
        double va1[][J1] = { {1,  1,  0},	
                             {1,  0,  1},
                             {0,  1,  1}               
                           };
        instance = affecterValeurs(I1, J1, &vc1[0], &va1[0][0]);
    }
    else if (numero == 2)
    {
        const int I2 = 7;
        const int J2 = 9;  
        double vc2[]     = { 10,  5,  8,  6,  9, 13, 11,  4,  6};	
        double va2[][J2] = { {1,  1,  1,  0,  1,  0,  1,  1,  0},	
                             {0,  1,  1,  0,  0,  0,  0,  1,  0},
                             {0,  1,  0,  0,  1,  1,  0,  1,  1},
                             {0,  0,  0,  1,  0,  0,  0,  0,  0},
                             {1,  0,  1,  0,  1,  1,  0,  0,  1},
                             {0,  1,  1,  0,  0,  0,  1,  0,  1},
                             {1,  0,  0,  1,  1,  0,  0,  1,  1}                 
                           };   
        instance = affecterValeurs(I2, J2, &vc2[0], &va2[0][0]);         
    }
        else if (numero == 3)
    {
        const int I3 = 10;
        const int J3 = 10;  
        double vc3[]     = {  7, 7, 5, 6, 3, 1, 3, 3, 7, 7};	
        double va3[][J3] = { {1, 1, 0, 0, 1, 1, 0, 1, 1, 0},
                             {0, 0, 1, 0, 1, 1, 0, 0, 0, 1},
                             {0, 1, 0, 0, 1, 0, 0, 1, 1, 1},
                             {1, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                             {0, 1, 1, 1, 1, 1, 0, 1, 0, 1},
                             {1, 0, 0, 0, 1, 0, 1, 1, 1, 1},
                             {1, 0, 1, 1, 0, 1, 1, 0, 1, 0},
                             {0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
                             {1, 1, 1, 0, 1, 0, 0, 0, 0, 0},
                             {0, 1, 1, 1, 1, 0, 0, 1, 1, 1}
                           };
        instance = affecterValeurs(I3, J3, &vc3[0], &va3[0][0]);                          
    }
    return instance;    
}

// ============================================================================
// parser d'instances de SPP

tInstance SPPparser(string fname)
{
    tInstance instance;
    int nbr1; // nombre de 1 sur la contrainte
    int posJ; // position d'un 1 sur la contrainte

    instance.fname = fname;

    ifstream fichier(fname, ios::in); 
    if(fichier)  // pas de probleme rencontre lors de l'ouverture du fichier
    {       
        // dimensions de l'instance
        fichier >> instance.I; 
        fichier >> instance.J; 

        // allocation dynamique
        instance.c = (double *) malloc( instance.J * sizeof(double) );

        instance.a = (double **) malloc( instance.I * sizeof(double *) );
        for(int i = 0; i < instance.I; i++)
            instance.a[i] = (double *) malloc( instance.J * sizeof(double) );

        instance.b = (double *) malloc( instance.I * sizeof(double) );


        // affectation des valeurs
        for (int j=0; j<instance.J; j++)
            fichier >> instance.c[j];

        for (int i=0; i<instance.I; i++)  
        {
            fichier >> nbr1;
            for (int v=0; v<nbr1; v++)
            {
                fichier >> posJ;
                instance.a[i][posJ-1] = 1;
            }
        }

        for (int i=0; i<instance.I; i++)   
            instance.b[i] = 1;  

        fichier.close();  
    }
    else  
        cerr << "Impossible d'ouvrir le fichier !" << endl;


    return instance;
}

// ============================================================================
// affichage de tous les parametres d'une instance numerique

void SPPafficheInstance(tInstance instance)
{
    // nom de l'instance  
    cout << "\nfname : " << instance.fname 
         << endl;

    // dimensions de l'instance  
    cout << "  (I) " << instance.I 
         << "  (J) " << instance.J 
         << endl << endl;

    // vecteur des couts  
    for (int j=0; j<instance.J; j++)
        cout << " " << instance.c[j] ;              
    cout << endl << endl;  

    // matrice des contraintes
    for (int i=0; i<instance.I; i++)
    {
        for (int j=0; j<instance.J; j++)
            cout << " " << instance.a[i][j] ;
        cout << endl;
    }
    cout << endl;                       
}


// ----------------------------------------------------------------------------
// Pose le modele correspondant a la relaxation lineaire du SPP 

GRBModel setupModel(GRBEnv& env, tInstance instance)
{
    int I = instance.I;
    int J = instance.J;
    double *  c = instance.c;	
    double ** a = instance.a;
    double *  b = instance.b;

    // Create an empty model
    GRBModel model = GRBModel(env);
    model.set(GRB_StringAttr_ModelName, "B&B");

    // Allocate dynamically the memory and create the variables ===============

    GRBVar* x = 0; 
    x = model.addVars(J);  // 0.0, 1.0, 0.0, GRB_BINARY, NULL, 7);   
    for (int j=0; j<J; j++) 
    {
        x[j].set(GRB_DoubleAttr_LB, 0.0);
        x[j].set(GRB_DoubleAttr_UB, 1.0);
        x[j].set(GRB_CharAttr_VType, GRB_CONTINUOUS);
        x[j].set(GRB_StringAttr_VarName, "x"+to_string(j+1));
    }                                                        

    // Set objective ==========================================================

    GRBLinExpr exprObj = 0;
    for (int j=0; j<J; j++) 
        exprObj += c[j] * x[j];
    model.setObjective(exprObj, GRB_MAXIMIZE);

    // Set constraints ========================================================

    for (int i=0; i<I; i++)
    {
        GRBLinExpr exprCte = 0;
        for (int j=0; j<J; j++)
            exprCte += a[i][j] * x[j];   
        model.addConstr(exprCte, GRB_LESS_EQUAL, b[i],  "cte_" + to_string(i+1));
    }

    model.write("BaBSPP.lp");

    return model;
}


// ----------------------------------------------------------------------------
// Affichage d'une solution (sur modele)

void displayResults(GRBModel& model)
{
    // nombre de variables dans le modele
    int nbrVar = model.get(GRB_IntAttr_NumVars);

    cout << "\nSolution Optimale :" << endl;
    cout << "  Obj= " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    
    GRBVar * vars = model.getVars() ;
    cout << "\nVariables" << endl;
    for (int j=0; j<nbrVar; j++)
    {
        cout << "  "
             << vars[j].get(GRB_StringAttr_VarName) 
             << " = "  
             << vars[j].get(GRB_DoubleAttr_X)
             << endl;
    }
}

// ----------------------------------------------------------------------------
// Affichage d'une solution (sur * modele)

void displayResults2(GRBModel * model)
{
    // nombre de variables dans le modele
    int nbrVar = model->get(GRB_IntAttr_NumVars);

    cout << "\nSolution Optimale :" << endl;
    cout << "  Obj= " << model->get(GRB_DoubleAttr_ObjVal) << endl;
    
    GRBVar * vars = model->getVars() ;
    cout << "\nVariables" << endl;
    for (int j=0; j<nbrVar; j++)
    {
        cout << "  "
             << vars[j].get(GRB_StringAttr_VarName) 
             << " = "  
             << vars[j].get(GRB_DoubleAttr_X)
             << endl;
    }
}


// ----------------------------------------------------------------------------
// Choix de la variable a separer; heuristique : valeur plus proche d'un entier

int heuristiqueSeparation(GRBModel& m, double tol)
{
    // nombre de variables dans le modele
    int nbrVar = m.get(GRB_IntAttr_NumVars);

    // pointeur sur les valeurs des variables
    double* values = NULL;
    values = m.get(GRB_DoubleAttr_X, m.getVars(), nbrVar);


    int    jPos = 0;
    double jVal = -1.0;

    for (int j=0; j<nbrVar; j++)
    {
        if ( (estApprox1(values[j],tol)==0) && (estApprox0(values[j],tol)==0) )
        {
            //cout <<"Heu:" << abs(0.5-values[j]) << " " << jVal << " ";
            if ( abs(0.5-values[j]) > jVal )
            {
                jVal =  abs(0.5-values[j]);
                jPos = j;
                cout << "best ->>";
            }
            cout << " frac : " << j << " --> " << values[j] << endl;
        }
    }
    cout << "jPos " << jPos << endl;

    return jPos;

}


// ----------------------------------------------------------------------------
// cree et traite un noeud fils d'un noeud actif

void infererNoeud(GRBModel& model, char DroiteGauche, int jPos, double tol, tSolution * bestSolution, list<tNoeud> * noeudsActifs, int * cptNoeuds)
{
    GRBVar * x = model.getVars() ;

    if (DroiteGauche=='D')
    {
        cout << "\nNOEUD DROIT ------------------------------------ " << endl;

        // fixe la variable en position jPos a 0
        cout << "\npose x[" << jPos << "]=0" << endl;
        x[jPos].set(GRB_DoubleAttr_LB, 0.0); // normalement pas necessaire
        x[jPos].set(GRB_DoubleAttr_UB, 0.0);  
    }
    else
    {
        cout << "\nNOEUD GAUCHE ----------------------------------- " << endl;
 
        // fixe la variable en position jPos a 1
        cout << "\npose x[" << jPos << "]=1" << endl;
        x[jPos].set(GRB_DoubleAttr_LB, 1.0);
        x[jPos].set(GRB_DoubleAttr_UB, 1.0); // normalement pas necessaire  
    }

    // optimise le modele 
    model.optimize();
    int status = model.get(GRB_IntAttr_Status);

    // tester si le noeud peut etre declare sonde

    // TEST 1 : sonde par non realisabilite ?
    if ((status == GRB_INF_OR_UNBD) ||
        (status == GRB_INFEASIBLE)  ||
        (status == GRB_UNBOUNDED)) 
    {
        cout << "sonde par non realisabilite => rien d'autre a faire" << endl;
    }
    // TEST 2 : sonde par optimalite ? 
    else if ( estSolution01(model, tol)==1 )
    {
        cout << "\nsonde par optimalite" << endl;
        // affiche le resultat de l'optimisation
        displayResults(model);

        // verifier si cette solution admissible entiere ameliore la borne primale
        if ( bestSolution->z < model.get(GRB_DoubleAttr_ObjVal) )
        {
            // meilleure borne primale
            cout << "Borne primale améliorée : passe de " << bestSolution->z << " a " << model.get(GRB_DoubleAttr_ObjVal) << endl; 
            bestSolution->z = model.get(GRB_DoubleAttr_ObjVal);
            bestSolution->x = clone_X(model);

            // verifier avec les sommets pendants si l'un ne peut pas être ferme car sondes par dominance
            // to do
        }         
    }
    // TEST 3 : sonde par dominance ?  
    /*      
    else if ( model.get(GRB_DoubleAttr_ObjVal) < bestSolution->z )
    {
        cout << "sonde par dominance => rien d'autre a faire" << endl;
    }
    */
    else if(floor(model.get(GRB_DoubleAttr_ObjVal))<=bestSolution->z){
        //cout << "sonde par dominance => couper " << endl;
    }
    // si aucun test declenche => noeud actif
    else
    {
        // affiche le resultat de l'optimisation 
        displayResults(model);
        model.update();

        // ajoute ce noeud comme noeud actif dans l'arboresence comme fils gauche ou droit
        tNoeud noeud;

        noeud.id         = *cptNoeuds;                       //  numero identifiant du noeud
        noeud.bDuale.z   = model.get(GRB_DoubleAttr_ObjVal); //  z pour relaxation lineaire
        noeud.bDuale.x   = clone_X(model);                   //  x pour relaxation lineaire
        noeud.model      = new GRBModel(model);              //  modele + variables fixees

        *cptNoeuds       = *cptNoeuds+1;

        noeudsActifs->push_front(noeud); // attention les info de resolution sont pas sauvegardees
    }
}


// ----------------------------------------------------------------------------
// teste si la solution est entiere (constituee que de valeurs 0 et 1)

int estSolution01(GRBModel&  m, double tol)
{

    // nombre de variables dans le modele
    int nbrVar = m.get(GRB_IntAttr_NumVars);

    // pointeur sur les valeurs des variables
    double* values = NULL;
    values = m.get(GRB_DoubleAttr_X, m.getVars(), nbrVar);

    int entier = 1; // booleen a vrai
    int j = 0;      // premier indice de la variable

    while ((entier==1) && (j<nbrVar))
    {
        if ( (estApprox1(values[j],tol)) || (estApprox0(values[j],tol)) )
            // vecteur x compose que de valeurs entieres, soit 0 soit 1
            j++;
        else
            // vecteur x compose d'au moins une valeur fractionnaire, la j-ieme
            entier = 0;
    }

    //if (entier==1)
    //    cout << "Solution entiere trouvee => sonde par optimalite" << endl;
    //else
    //    cout << "Solution non-entiere : x[" << j+1 << "]=" << values[j] << endl;

    return entier;
}


// ----------------------------------------------------------------------------
// approximation pour tester l'egalite d'un 0.0 flottant

int estApprox0(double x, double tol)
{
    return (abs(x) <= tol);
}


// ----------------------------------------------------------------------------
int estApprox1(double x, double tol)
// approximation pour tester l'egalite d'un 1.0 flottant
{
    return (abs(1.0 - x) <= tol);
}


// ----------------------------------------------------------------------------
// allocation dynamique de memoire et copie les valeurs d'une solution 

double * clone_X(GRBModel&  m)
{
    // nombre de variables dans le modele
    int nbrVar = m.get(GRB_IntAttr_NumVars);

    // pointeur sur les valeurs des variables
    double* values = NULL;
    values = m.get(GRB_DoubleAttr_X, m.getVars(), nbrVar);

    // cloning...
    double *   x = (double *) malloc(nbrVar * sizeof(double));
    for (int i=0; i<nbrVar; i++)
    {
        x[i] = values[i];
        //cout << " --> " <<  i << " " << x[i] << endl;
    }
    return x;
}


// ----------------------------------------------------------------------------
// libere proprement la memoire allouee dynamiquement pour une instance

void SPPlibereMemoire(tInstance instance)
{
    delete instance.c;

    for (int i=0; i<instance.I; i++)  
        delete instance.a[i];

    delete instance.b;
}