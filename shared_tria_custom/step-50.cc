/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Thomas C. Clevenger, Clemson University
 *         Timo Heister, Clemson University
 *         Guido Kanschat, Heidelberg University
 *         Martin Kronbichler, Technical University of Munich
 */




#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>


using namespace dealii;


template <int dim>
void mypartition(parallel::shared::Triangulation<dim> &tria)
{
  int n_procs = 4;

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcNotImplemented());
  //  int n_subdomains = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  for (auto &cell : tria.active_cell_iterators())
    cell->set_subdomain_id(3);

  if (false)
  {
    GridTools::partition_triangulation_zorder(n_procs, tria);
  }
  else
  {
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active();
    for (unsigned int i=0; i < tria.n_active_cells(); ++i)
      {
        unsigned int j=0;
        if (i<4)
          j=1;
        else if (i<8)
          j=2;

        cell->set_subdomain_id(j % n_proc);
        ++cell;
      }
  }

  GridTools::partition_multigrid_levels(tria);
}


template <int dim>
class LaplaceProblem
{
public:
  LaplaceProblem();
  void run();

private:
  void setup_system();
  void refine_grid();
  void output_results(const unsigned int cycle);

  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  parallel::shared::Triangulation<dim> shared_tria;
  //parallel::shared::Triangulation<dim> p4est_tria;
};


template <int dim>
LaplaceProblem<dim>::LaplaceProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))

  , shared_tria(mpi_communicator,
                typename Triangulation<dim>::MeshSmoothing
                (Triangulation<dim>::limit_level_difference_at_vertices),
                true,
                typename parallel::shared::Triangulation<dim>::Settings
                (parallel::shared::Triangulation<dim>::partition_custom_signal))

  //  , p4est_tria(mpi_communicator,
  //               typename Triangulation<dim>::MeshSmoothing
  //               (Triangulation<dim>::limit_level_difference_at_vertices),
  //               true,
  //               typename parallel::shared::Triangulation<dim>::Settings
  //               (parallel::shared::Triangulation<dim>::partition_zorder |
  //                parallel::shared::Triangulation<dim>::construct_multigrid_hierarchy))
{
  shared_tria.signals.post_refinement.connect (std::bind(&mypartition<dim>, std::ref(shared_tria)));

  GridGenerator::hyper_cube(shared_tria, -1., 1., /*colorize*/ false);
  //shared_tria.refine_global(1);

  //  GridGenerator::hyper_cube(p4est_tria, -1., 1., /*colorize*/ false);
  //  p4est_tria.refine_global(1);
}



template <int dim>
void LaplaceProblem<dim>::setup_system()
{

}



template <int dim>
void LaplaceProblem<dim>::refine_grid()
{
  shared_tria.refine_global();
  return;

  for (auto cell : shared_tria.active_cell_iterators())
    if (cell->center()[0] < 0 && cell->center()[1] < 0)
      cell->set_refine_flag();
  shared_tria.execute_coarsening_and_refinement();

  //  for (auto cell : p4est_tria.active_cell_iterators())
  //    if (cell->center()[0] < 0 && cell->center()[1] < 0)
  //      cell->set_refine_flag();
  //  p4est_tria.execute_coarsening_and_refinement();
}



template <int dim>
void LaplaceProblem<dim>::output_results(const unsigned int cycle)
{
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(shared_tria,
                                           "shared_tria_active"+Utilities::int_to_string(cycle),
                                           false,
                                           false);
  //  grid_out.write_mesh_per_processor_as_vtu(p4est_tria,
  //                                           "p4est_tria_active"+Utilities::int_to_string(cycle),
  //                                           false,
  //                                           false);
}



template <int dim>
void LaplaceProblem<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 3; ++cycle)
  {
    std::cout << "cycle: " << cycle << std::endl;
    if (cycle > 0)
      refine_grid();

    output_results(cycle);

    std::cout << std::endl;
  }
}



int main(int argc, char *argv[])
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
  {
    LaplaceProblem<2> test;
    test.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 1;
  }

  return 0;
}
