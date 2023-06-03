#include <iostream>
#include <boost/version.hpp>
#include <armadillo>



//
//using namespace arma;


int main() {
    std::cout << "Boost version: " << BOOST_LIB_VERSION << std::endl;
    arma::vec v(3); // Crear un vector de Armadillo

    v << 1 << 2 << 3; // Asignar valores al vector

    std::cout << "Vector: " << v << std::endl; // Imprimir el vector

    return 0;
}