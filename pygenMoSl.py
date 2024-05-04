import numpy as np
import mendeleev
from ase.db import connect
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN

class Gen_MoSl_descritptor(object):
    def sort_descriptor(func):
        def wrapper(self, *args, **kwargs):
            descriptor = func(self, *args, **kwargs)
            sorted_descriptor = []
            for i in range(int(len(descriptor)/2)):
                sorted_descriptor.extend(sorted(descriptor[i*2: i*2+2], reverse=False, key=lambda x: x[0]))
            return sorted_descriptor
        return wrapper
    
    def __init__(self, atoms):
        self.atoms = atoms
        elements = list(set([i.symbol for i in self.atoms if i.symbol != 'O'])) #expect adsorbate O

        self.elements_ = elements
        mendeleev_data = dict.fromkeys(elements)
        for element in mendeleev_data:
            mendeleev_data[element] = getattr(mendeleev, element)
        self.mendeleev_data_ = mendeleev_data
        elements.sort(key=lambda x: self.mendeleev_data_[x].atomic_number)
        self.elements_ = elements
        self.lattice_const = {'Ag':_user-defined_, 'Au':_user-defined_, 'Cu':_user-defined_, 'Ir':_user-defined_, 
                              'Pd':_user-defined_, 'Pt':_user-defined_, 'Rh':_user-defined_, 'Ru':_user-defined_}
        
        self.ad_data = {'Ag':_user-defined_, 'Au':_user-defined_, 'Cu':_user-defined_, 'Ir':_user-defined_, 
                        'Pd':_user-defined_, 'Pt':_user-defined_, 'Rh':_user-defined_, 'Ru':_user-defined_}
        self.avg_ad_energy_ = self.ad_data
    
    def split_atoms_by_layer(self, z_tol=0.5):
        """This function splits the atoms from an ASE Atoms object into separate layers, with each layer defined as being within the given tolerance of one another.

        Parameters: 
            atoms (ase.Atoms): Atoms object to be split into layers
            tolerance (float): Tolerance value for layering

        Returns:
            list of lists containing Atoms objects for each layer found
        """
        atoms = self.atoms.copy()
        del atoms[[i.index for i in atoms if i.symbol == _user-defined_ ]]
        
        # Create empty list to store layers
        layers = []

        # Loop over all atoms and save their z positions
        z_pos = []
        for atom in atoms:
            z_pos.append(atom.position[2])

        # Sort the z positions from lowest to highest
        z_pos.sort(reverse=True)

        # Iterate over the sorted z positions and divide them into layers when the difference between two is greater than the tolerance
        curr_layer = []
        curr_width = 0.0
        for pos in z_pos:
            if curr_width == 0.0:
                curr_width = pos
            else:
                if curr_width - pos < z_tol:
                    curr_layer.append(pos)
                else:
                    layers.append(curr_layer)
                    curr_layer = [pos]
                    curr_width = pos
        # Append the final layer
        layers.append(curr_layer)

        # Create empty list stores the layer's atoms
        out_list = []
        for layer in layers:
            # Iterate the Atoms object and check which ones are inside the layer boundary
            layer_atoms = []
            for atom in atoms:
                if atom.position[2] <= layer[0] and atom.position[2] >= layer[-1]:
                    layer_atoms.append(atom)
            # Save the atom list for each layer
            out_list.append(layer_atoms)
            
        assert len(out_list) == 5, 'Error! The number of layers is not 5'
        for i in out_list:
            assert len(i) == 12, 'Error! The number of atoms in each layer is not 12'
        
        self.layers_ = out_list
        return (self.layers_)
    
    def get_elements_from_atoms(self, atoms_lst):
        elements = list(set([i.symbol for i in atoms_lst]))
        elements.sort(key=lambda x: self.mendeleev_data_[x].atomic_number)
        return elements
    
    def get_dummy_features(self):
        candidate_elements = list(self.ad_data.keys())
        atomic_number = np.mean([self.lattice_const[i] for i in candidate_elements])
        electronegativity = np.mean([getattr(mendeleev, i).electronegativity(scale='mulliken') 
                                     for i in candidate_elements])
        element_avg_ad_energy = np.mean([self.ad_data[i] for i in candidate_elements])
        dummy_features = [atomic_number, electronegativity, element_avg_ad_energy, element_count]
        return dummy_features
    
    @sort_descriptor
    def gen_layers_descriptor(self):
        descriptor_layers = []
        for layer in self.layers_:
            descriptor_layer = dict.fromkeys(self.get_elements_from_atoms(layer))
            for element in descriptor_layer.keys():
                descriptor_layer[element] = [self.lattice_const[element],
                                             self.mendeleev_data_[element].electronegativity(scale='mulliken'),
                                             self.avg_ad_energy_[element], 0]
                #atomic number, electronegativity, element_avg_ad_energy, element_count
            for i in layer:
                descriptor_layer[i.symbol][3] += 1
            if len(descriptor_layer.keys()) == 1:
                descriptor_layer['dummy'] = self.get_dummy_features()

            descriptor_layers.extend(list(descriptor_layer.values()))
        return descriptor_layers       
    
    @staticmethod
    def __get_coordination_string(nn_info:str) -> str:
        coordinated_atoms = [neighbor_info['site'].species_string
                            for neighbor_info in nn_info
                            if neighbor_info['site'].species_string != 'O']
        coordination = '-'.join(sorted(coordinated_atoms))
        return coordination
    
    def fingerprint_adslab(self):
        oxygen_index = self.atoms.get_chemical_symbols().index('O')
        struct = AseAtomsAdaptor.get_structure(self.atoms)
        try:
            # We have a standard and a loose Voronoi neighbor finder for various
            # purposes
            vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=10)

            # Find the coordination
            nn_info = vnn.get_nn_info(struct, n=oxygen_index)
            coordination = self.__get_coordination_string(nn_info)

            # Find the neighborcoord
            neighborcoord = []
            for neighbor_info in nn_info:
                # Get the coordination of this neighbor atom, e.g., 'Cu-Cu'
                neighbor_index = neighbor_info['site_index']
                neighbor_nn_info = vnn_loose.get_nn_info(struct, n=neighbor_index)
                neighbor_coord = self.__get_coordination_string(neighbor_nn_info)
                # Prefix the coordination of this neighbor atom with the identity
                # of the neighber, e.g. 'Cu:Cu-Cu'
                neighbor_element = neighbor_info['site'].species_string
                neighbor_coord_labeled = neighbor_element + ':' + neighbor_coord
                neighborcoord.append(neighbor_coord_labeled)

            # Find the nextnearestcoordination
            nn_info_loose = vnn_loose.get_nn_info(struct, n=oxygen_index)
            nextnearestcoordination = self.__get_coordination_string(nn_info_loose)

            return {'coordination': coordination,
                    'neighborcoord': neighborcoord,
                    'nextnearestcoordination': nextnearestcoordination}
        except:
            print ('Error! VoronoiNN failed to find coordination')
    
    @staticmethod
    def inner_shell(doc):
        inner_shell_atoms = doc['coordination'].split('-')

        # Sometimes there is no coordination. If this happens, then hackily reformat it
        if inner_shell_atoms == ['']:
            inner_shell_atoms = []

        return inner_shell_atoms

    @staticmethod
    def outter_shell(doc):
        outter_shell_atoms = []
        for neighbor_coordination in doc['neighborcoord']:
            _, coordination = neighbor_coordination.split(':')
            coordination = coordination.split('-')
            outter_shell_atoms.extend(coordination)
        return outter_shell_atoms    
    
    @sort_descriptor
    def gen_shell_descriptor(self, shell_list):
        shell_elements = list(set(shell_list))
        # shell_elements.sort(key=lambda x: self.mendeleev_data_[x].atomic_number)
        shell_descriptor = {}
        for element in shell_elements:
            shell_descriptor[element] = [self.lattice_const[element],
                            self.mendeleev_data_[element].electronegativity(scale='mulliken'),
                            self.avg_ad_energy_[element], shell_list.count(element)]
        if len(shell_descriptor.keys()) == 1:
            shell_descriptor['dummy'] = self.get_dummy_features()
        return list(shell_descriptor.values())

    @sort_descriptor     
    def gen_motif_descriptor(self):
        inner_shell_atoms = self.inner_shell(self.fingerprint_adslab())
        assert len(inner_shell_atoms) == 3, \
            f'Error! The coordination number of O is {len(inner_shell_atoms)}, not 3!'
        
        outter_shell_atoms = self.outter_shell(self.fingerprint_adslab())
        inner_shell_descriptor = self.gen_shell_descriptor(inner_shell_atoms)
        outter_shell_descriptor = self.gen_shell_descriptor(outter_shell_atoms)
        motif_descriptor = inner_shell_descriptor + outter_shell_descriptor
        return motif_descriptor

    @sort_descriptor
    def gen_descriptor(self):
        motif_descriptor = self.gen_motif_descriptor()
        descriptor_layers = self.gen_layers_descriptor()
        descriptor = motif_descriptor + descriptor_layers
        return descriptor
    
    @sort_descriptor
    def gen_descriptor1(self):
        motif_descriptor = np.array(self.gen_motif_descriptor())
        descriptor_layers = np.array(self.gen_layers_descriptor())
        descriptor = motif_descriptor @ descriptor_layers.T
        return descriptor
    
if __name__ == '__main__':
    import ase
    atoms = ase.io.read(_user-defined_)
    dcpt = Gen_MoSl_descritptor(atoms)
    dcpt.split_atoms_by_layer()
    MoSl_descriptor = dcpt.gen_descriptor()
