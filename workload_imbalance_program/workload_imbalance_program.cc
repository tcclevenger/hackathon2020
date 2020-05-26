/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *         Timo Heister, University of Goettingen, 2009, 2010
 */


#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <fstream>
#include <iostream>

bool martin = false;


namespace Step40
{
using namespace dealii;


template <int dim>
class LaplaceProblem
{
public:
  LaplaceProblem ();
  ~LaplaceProblem ();

  void run ();

private:
  void refine_grid (std::string grid_type);
  void generate_mesh(std::string grid_type, unsigned int global_refinements = 0, unsigned int n_subdiv = 0);

  void ownership_data (unsigned int n_procs);
  void distribute_mesh(unsigned int n_procs);

  MPI_Comm                                  mpi_communicator;

  parallel::shared::Triangulation<dim,dim>  triangulation;

  ConditionalOStream                        pcout;
};





template <int dim>
LaplaceProblem<dim>::LaplaceProblem ()
  :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::limit_level_difference_at_vertices |
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening), false),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0))
{
}



template <int dim>
LaplaceProblem<dim>::~LaplaceProblem ()
{
}


template <int dim>
void LaplaceProblem<dim>::refine_grid (std::string grid_type)
{
  if (grid_type == "global-2d" || grid_type ==  "global-3d")
  {
    triangulation.refine_global();
  }
  else if (grid_type == "quadrant-2d")
  {
    typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

    for (; cell!=endc; ++cell)
    {
      bool first_quadrant = true;
      for (int d=0; d<dim; ++d)
        if (cell->center()[d]>0.0)
          first_quadrant = false;
      if (!first_quadrant)
        continue;
      cell->set_refine_flag();
    }
    triangulation.execute_coarsening_and_refinement ();
  }
  else if (grid_type == "circle-2d")
  {
    typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
    for (; cell != endc; ++cell)
      for (unsigned int vertex=0;
           vertex < GeometryInfo<dim>::vertices_per_cell;
           ++vertex)
      {
        {
          const Point<dim> p = cell->vertex(vertex);
          const Point<dim> origin = (dim == 2 ?
                                       Point<dim>(0,0) :
                                       Point<dim>(0,0,0));
          const double dist = p.distance(origin);
          if (dist<0.25/M_PI)
          {
            cell->set_refine_flag ();
            break;
          }
        }
      }
    triangulation.execute_coarsening_and_refinement ();
  }
  else if (grid_type == "sine_curve")
  {
    for (auto cell : triangulation.active_cell_iterators())
    {
      bool above = false;
      bool below = false;
      bool left = false;
      bool right = false;

      for (unsigned int v=0;
           v < GeometryInfo<dim>::vertices_per_cell;
           ++v)
      {
        const double x_value = cell->vertex(v)[0];
        const double y_value = cell->vertex(v)[1];
        double z_value = (dim==3 ? cell->vertex(v)[2] : 0);

        const double sin_value = 0.75*std::sin(3.0*x_value-2.5)+0.2;

        if (dim == 2)
        {
          if (y_value <= sin_value)
            below = true;
          if (y_value >= sin_value)
            above = true;
        }
        else if (dim ==3)
        {
          if (z_value <= sin_value)
            below = true;
          if (z_value >= sin_value)
            above = true;

          //                    if (y_value <= sin_value)
          //                        left = true;
          //                    if (y_value >= sin_value)
          //                        right = true;
        }

        if (dim == 2 && above && below)
        {
          cell->set_refine_flag ();
          break;
        }
        if (dim == 3 && above && below)// && left && right)
        {
          cell->set_refine_flag ();
          break;
        }
      }
    }

    triangulation.execute_coarsening_and_refinement();
  }
}

template <int dim>
void LaplaceProblem<dim>::generate_mesh (std::string grid_type, unsigned int global_refinements, unsigned int n_subdiv)
{
  if (grid_type == "global-2d" || grid_type == "quadrant-2d" || grid_type == "circle-2d" || grid_type == "sine_curve")
  {
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global(global_refinements);
  }
  else if (grid_type == "global-3d")
  {
    unsigned int n_subdiv = 1;
    GridGenerator::subdivided_hyper_cube (triangulation, n_subdiv, -1, 1);
    triangulation.refine_global(global_refinements);
  }
  else if (grid_type == "annulus")
  {
    GridGenerator::subdivided_hyper_cube (triangulation, n_subdiv, -1, 1);
    triangulation.refine_global(global_refinements);
    if (true)
    {
      for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned() &&
            cell->center().norm() < 0.55)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
      for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned() &&
            cell->center().norm() > 0.3 && cell->center().norm() < 0.42)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
      for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned() &&
            cell->center().norm() > 0.335 && cell->center().norm() < 0.39)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }
  }
}



/*
  * This function creates data tables for ownership/ghost
  * owner data
  */
template <int dim>
void LaplaceProblem<dim>::ownership_data (unsigned int n_procs)
{
  unsigned int n_levels = triangulation.n_levels();
  unsigned int n_vertices = triangulation.n_vertices();

  std::vector<unsigned int> global_n_owned_cells_active(n_procs);
  std::vector<unsigned int> global_n_owned_cells_level(n_procs * n_levels);
  {
    typename Triangulation<dim>::cell_iterator
        cell = triangulation.begin(),
        endc = triangulation.end();
    for (; cell!=endc; ++cell)
    {
      if (cell->is_active())
        global_n_owned_cells_active[cell->subdomain_id()] += 1;
      global_n_owned_cells_level[cell->level() + cell->level_subdomain_id()*n_levels] += 1;
    }
  }

  //Ghost Data
  //Active Cells
  std::vector<std::set<unsigned int> > vertex_owners_active (n_vertices);
  {
    typename parallel::shared::Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(), endc = triangulation.end();
    for (; cell != endc; ++cell)
    {
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        vertex_owners_active[cell->vertex_index(v)].insert(cell->subdomain_id());
      }
    }
  }

  std::vector<std::set<unsigned int> > ghost_owners_active_per_proc(n_procs);
  std::vector<std::set<unsigned int> > total_ghost_owners_per_proc(n_procs);

  for (unsigned int v=0; v<vertex_owners_active.size(); ++v)
  {
    if (vertex_owners_active[v].size() == 1)
      continue;
    for (std::set<unsigned int>::iterator it_ex=vertex_owners_active[v].begin();
         it_ex!=vertex_owners_active[v].end(); ++it_ex)
    {
      for (std::set<unsigned int>::iterator it_in=vertex_owners_active[v].begin();
           it_in!=vertex_owners_active[v].end(); ++it_in)
      {
        if (*it_ex==*it_in)
          continue;
        ghost_owners_active_per_proc[*it_ex].insert(*it_in);
        total_ghost_owners_per_proc[*it_ex].insert(*it_in);
      }
    }
  }

  //Level cells
  // Step1: Add neighbors on same level
  std::vector<std::set<unsigned int> > ghost_owners_level_per_proc(n_procs*n_levels);
  for (unsigned int lvl=0; lvl<n_levels; ++lvl)
  {
    std::vector<std::set<unsigned int> > vertex_owners_level (n_vertices);//*n_levels);
    typename parallel::shared::Triangulation<dim>::cell_iterator
        cell = triangulation.begin(lvl), endc = triangulation.end(lvl);
    for (; cell != endc; ++cell)
    {
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        vertex_owners_level[cell->vertex_index(v)].insert(cell->level_subdomain_id());
      }
    }

    for (unsigned int v=0; v<vertex_owners_level.size(); ++v)
    {
      if (vertex_owners_level[v].size() == 1)
        continue;
      for (std::set<unsigned int>::iterator it_ex=vertex_owners_level[v].begin();
           it_ex!=vertex_owners_level[v].end(); ++it_ex)
      {
        for (std::set<unsigned int>::iterator it_in=vertex_owners_level[v].begin();
             it_in!=vertex_owners_level[v].end(); ++it_in)
        {
          if (*it_ex==*it_in)
            continue;
          ghost_owners_level_per_proc[lvl + (*it_ex)*n_levels].insert(*it_in);
          total_ghost_owners_per_proc[(*it_ex)].insert(*it_in);
        }
      }
    }
  }

  //Step 2: Add parent of owned cells
  for (unsigned int lvl=0; lvl<n_levels-1; ++lvl)
  {
    typename parallel::shared::Triangulation<dim>::cell_iterator
        cell = triangulation.begin(lvl), endc = triangulation.end(lvl);
    for (; cell != endc; ++cell)
    {
      if (!cell->has_children())
        continue;

      for (unsigned int c=0; c<GeometryInfo<dim>::max_children_per_cell; ++c)
      {
        if (cell->child(c)->level_subdomain_id() == cell->level_subdomain_id())
          continue;
        ghost_owners_level_per_proc[lvl + cell->child(c)->level_subdomain_id()*n_levels]
            .insert(cell->level_subdomain_id());
        total_ghost_owners_per_proc[cell->child(c)->level_subdomain_id()]
            .insert(cell->level_subdomain_id());


      }
    }
  }

  //Step 3: Neighbors on active level
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
      vertex_owners_active_for_level(n_vertices);
  {
    typename parallel::shared::Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(), endc = triangulation.end();
    for (; cell != endc; ++cell)
    {
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        vertex_owners_active_for_level[cell->vertex_index(v)]
            .push_back(std::make_pair(cell->level(),cell->subdomain_id()));
      }
    }
  }

  for (unsigned int v=0; v<vertex_owners_active_for_level.size(); ++v)
  {
    if (vertex_owners_active_for_level[v].size() == 1)
      continue;
    for (std::vector<std::pair<unsigned int, unsigned int>>::iterator
         it_ex=vertex_owners_active_for_level[v].begin();
         it_ex!=vertex_owners_active_for_level[v].end(); ++it_ex)
    {
      for (std::vector<std::pair<unsigned int, unsigned int>>::iterator
           it_in=vertex_owners_active_for_level[v].begin();
           it_in!=vertex_owners_active_for_level[v].end(); ++it_in)
      {
        if ((*it_ex).second == (*it_in).second)
          continue;
        if ((*it_ex).first > (*it_in).first)
        {
          ghost_owners_level_per_proc[(*it_ex).first + (*it_in).second * n_levels]
              .insert((*it_ex).second);
          total_ghost_owners_per_proc[(*it_in).second]
              .insert((*it_ex).second);
        }
      }
    }
  }


  // Print ratio/max ghost to screen
  unsigned long long int work_estimate = 0;
  unsigned long long int total_cells_in_hierarchy = 0;
  unsigned int max_ghost_owners = 0;
  for (unsigned int j=0; j<n_levels; ++j)
  {
    unsigned long long int max_on_lvl_j = 0;
    unsigned long long int total_cells_on_lvl = 0;

    for (unsigned int i=0; i<n_procs; ++i)
    {
      total_cells_on_lvl += global_n_owned_cells_level[j+i*n_levels];
      if (global_n_owned_cells_level[j+i*n_levels] > max_on_lvl_j)
        max_on_lvl_j = global_n_owned_cells_level[j+i*n_levels];

      if (ghost_owners_level_per_proc[j + i*n_levels].size() > max_ghost_owners)
        max_ghost_owners = ghost_owners_level_per_proc[j + i*n_levels].size();
    }
    work_estimate += max_on_lvl_j;
    total_cells_in_hierarchy += total_cells_on_lvl;
  }
  double ideal_work = total_cells_in_hierarchy / (double)n_procs;
  double workload_imbalance_ratio = work_estimate / ideal_work;


  // communication ratio
  unsigned int children_foreign=0;
  unsigned int children_native=0;
  double comm_ratio=0.0;

  typename parallel::shared::Triangulation<dim>::cell_iterator
      cell = triangulation.begin(), endc = triangulation.end();
  for (; cell != endc; ++cell)
  {
    if (!cell->has_children())
      continue;

    const unsigned int cell_owner = cell->level_subdomain_id();

    for (unsigned int c=0; c<GeometryInfo<dim>::max_children_per_cell; ++c)
    {
      if (cell->child(c)->level_subdomain_id() == cell_owner)
        ++children_native;
      else
        ++children_foreign;
    }
  }
  comm_ratio = (double)children_foreign / (double)(children_foreign + children_native);


  pcout << "Cores: " << n_procs << " "
        << "Ratio: " << workload_imbalance_ratio << " (" << 1.0/workload_imbalance_ratio <<") "
        << "Comm: " << comm_ratio << " (n: " << children_native << ", f: " << children_foreign << ") "
        << "Max ghost owners: " << max_ghost_owners << std::endl;
}


template <int dim>
void LaplaceProblem<dim>::distribute_mesh (unsigned int n_procs)
{
  //GridTools::partition_triangulation_zorder(n_procs, triangulation);

  for (int level=triangulation.n_global_levels()-1; level>=0; --level)
  {
    unsigned int n_level_cells = 0;
    for (auto cell : triangulation.active_cell_iterators_on_level(level))
    {
      (void)cell;
      n_level_cells += 1;
    }

    const unsigned int cells_per_proc = std::ceil((double)n_level_cells/(double)n_procs);
    unsigned int current_cells = 0;
    int current_proc = 0;
    for (auto cell : triangulation.active_cell_iterators_on_level(level))
    {
      cell->set_subdomain_id(current_proc);
      current_cells += 1;

      if (current_cells >= cells_per_proc)
      {
        current_cells = 0;
        current_proc += 1;
      }
    }
  }

  GridTools::partition_multigrid_levels(triangulation);
}


template <int dim>
void LaplaceProblem<dim>::run ()
{

  std::string grid_type = "";

  if (martin)
  {
    grid_type = "annulus";
    pcout << "Annulus Refinement " << dim << "D" << std::endl;
    for (unsigned int procs = 16; procs <66000; procs*=2)
    {
      pcout << "Number of cores: " << procs << std::endl;
      unsigned int sizes [] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128};
      unsigned int n_cycles = sizeof(sizes)/sizeof(unsigned int);
      for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        unsigned int n_refinements = 0;
        unsigned int n_subdiv = sizes[cycle];
        if (n_subdiv > 1)
          while (n_subdiv%2 == 0)
          {
            n_refinements += 1;
            n_subdiv /= 2;
          }
        if (dim == 2)
          n_refinements += 3;
        unsigned int njobs = procs;
        while (njobs > 0)
        {
          njobs >>= dim;
          n_refinements++;
        }
        triangulation.clear();
        generate_mesh(grid_type,n_refinements,n_subdiv);

        if (dim == 2 && triangulation.n_active_cells()/(double)(procs) > 6e5)
          break;
        if (dim == 3 && triangulation.n_active_cells()/(double)(procs) > 4e5)
          break;
        if (triangulation.n_active_cells()/(double)(procs) < 1e3)
          continue;

        distribute_mesh(procs);
        ownership_data (procs);


        if (dim == 2 && triangulation.n_active_cells() > 2.7e8)
          break;
        if (dim == 3 && triangulation.n_active_cells() > 1.3e8)
          break;
      }
      pcout << std::endl;
    }
    pcout << std::endl << std::endl;
  }
  else
  {
    if (dim == 2)
    {
      //      grid_type = "global-2d";
      //      pcout << "Global Refinement " << dim << "D" << std::endl;
      //      for (unsigned int cycle=0; cycle<13; ++cycle)
      //      {
      //        if (cycle == 0)
      //        {
      //          triangulation.clear();
      //          generate_mesh(grid_type,0);
      //        }
      //        else
      //          refine_grid (grid_type);

      //        distribute_mesh(4);

      //        GridOut grid_out;
      //        grid_out.write_mesh_per_processor_as_vtu(triangulation,
      //                                                 "active-mesh"+Utilities::int_to_string(cycle),
      //                                                 false,
      //                                                 false);
      //        grid_out.write_mesh_per_processor_as_vtu(triangulation,
      //                                                 "level-mesh"+Utilities::int_to_string(cycle),
      //                                                 true,
      //                                                 false);

      //        if (cycle > 4)
      //          break;

      //        if (cycle < 6)
      //          continue;

      //        pcout << "Active cells: " << triangulation.n_global_active_cells() << std::endl;

      //        for (unsigned int procs = 16; procs <2e6; procs*=2)
      //        {
      //          if (triangulation.n_global_active_cells()/procs < 100 ||
      //              triangulation.n_global_active_cells()/procs > 500e3)
      //            continue;

      //          distribute_mesh(procs);
      //          ownership_data (procs);
      //        }
      //        pcout << std::endl;
      //      }

      grid_type = "quadrant-2d";
      pcout << "Quadrent Refinement " << dim << "D" << std::endl;
      for (unsigned int cycle=0; cycle<12; ++cycle)
      {
        //pcout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
          triangulation.clear();
          generate_mesh(grid_type,2);
        }
        else
          refine_grid (grid_type);

        pcout << "Active cells: " << triangulation.n_global_active_cells() << std::endl;

        for (unsigned int procs = 16; procs <2e6; procs*=2)
        {
          if (triangulation.n_global_active_cells()/procs < 100 ||
              triangulation.n_global_active_cells()/procs > 500e3)
            continue;

          distribute_mesh(procs);
          ownership_data (procs);
        }
        pcout << std::endl;
      }

    //      grid_type = "circle-2d";
    //      pcout << "Circle Refinement " << dim << "D" << std::endl;
    //      for (unsigned int procs = 16; procs <33000; procs*=2)
    //      {
    //        pcout << "Number of cores: " << procs << std::endl;
    //        for (unsigned int cycle=0; cycle<14; ++cycle)
    //        {
    //          //pcout << "Cycle " << cycle << ':' << std::endl;
    //          if (cycle == 0)
    //          {
    //            triangulation.clear();
    //            generate_mesh(grid_type,3);
    //          }
    //          else
    //            refine_grid (grid_type);

    //          if (cycle < 8)
    //            continue;
    //          distribute_mesh(procs);
    //          ownership_data (procs);
    //        }
    //        pcout << std::endl;
    //      }

    //            grid_type = "sine_curve";
    //            pcout << "Sine Refinement " << dim << "D" << std::endl;
    //            for (unsigned int cycle=0; cycle<21; ++cycle)
    //            {
    //                Timer timer;
    //                timer.restart();
    //                if (cycle == 0)
    //                {
    //                    triangulation.clear();
    //                    generate_mesh(grid_type,2);
    //                }
    //                else
    //                    refine_grid (grid_type);
    //                timer.stop();
    //                // std::cout << timer.last_cpu_time() << std::endl;

    //                pcout << "Active cells: " << triangulation.n_global_active_cells() << std::endl;

    //                //                    if (cycle < 15)
    //                //                        continue;

    //                for (unsigned int procs = 16; procs <2e6; procs*=2)
    //                {
    //                    if (triangulation.n_global_active_cells()/procs < 100 ||
    //                        triangulation.n_global_active_cells()/procs > 500e3)
    //                        continue;

    //                    distribute_mesh(procs);
    //                    ownership_data (procs);
    //                }
    //                pcout << std::endl;
    //            }
    pcout << std::endl << std::endl;
  }
  else if (dim==3)
  {
    //            grid_type = "sine_curve";
    //            pcout << "Sine Refinement " << dim << "D" << std::endl;
    //            for (unsigned int cycle=0; cycle<11; ++cycle)
    //            {
    //                Timer timer;
    //                timer.restart();
    //                if (cycle == 0)
    //                {
    //                    triangulation.clear();
    //                    generate_mesh(grid_type,1);
    //                }
    //                else
    //                    refine_grid (grid_type);
    //                timer.stop();
    //                // std::cout << timer.last_cpu_time() << std::endl;

    //                pcout << "Active cells: " << triangulation.n_global_active_cells() << std::endl;

    //                //                    if (cycle < 15)
    //                //                        continue;

    //                for (unsigned int procs = 16; procs <33e3; procs*=2)
    //                {
    //                    //unsigned int procs = 4;
    //                    if (triangulation.n_global_active_cells()/procs < 100 ||
    //                            triangulation.n_global_active_cells()/procs > 500e3)
    //                        continue;

    //                    distribute_mesh(procs);
    //                    ownership_data (procs);

    //                    GridOut grid_out;
    //                    grid_out.write_mesh_per_processor_as_vtu(triangulation,
    //                                                             "active-mesh"+Utilities::int_to_string(cycle),
    //                                                             false,
    //                                                             false);
    //                    grid_out.write_mesh_per_processor_as_vtu(triangulation,
    //                                                             "level-mesh"+Utilities::int_to_string(cycle),
    //                                                             true,
    //                                                             false);
    //               }
    //                pcout << std::endl;
    //            }
  }
  pcout << std::endl << std::endl;
}
}
}




int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);


  if (argc == 2)
  {
    martin = true;
  }
  else
  {
    martin = false;
  }

  try
  {
    using namespace dealii;
    using namespace Step40;


    {
      LaplaceProblem<2> laplace_problem_2d;
      laplace_problem_2d.run ();
    }
    {
      LaplaceProblem<3> laplace_problem_2d;
      laplace_problem_2d.run ();
    }


  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
