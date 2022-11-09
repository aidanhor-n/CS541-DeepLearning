"""VirtualPowder
This base class is used to generate virtual powders given source powders to mix and their percent composition.

There are numerous extensions of this class for different solutions on sampling and flowability prediction

ASSUMPTIONS: all collected source powders have a mass of 50g

Sampling:
VirtualPowder (Base Class): Naive percentage of mass sampling (should be accurate enough)
TODO VirtualPowderPercentage: Ignore mass equality assumption and create pure percentage based particle mix in virtual powder
TODO VirtualPowderMass: Utilize volume feature to assume density and create balanced powder mix by mass

Flowability Prediction:
VirtualPowder (Base Class): Naive percentage component sum flowability prediction
TODO VirtualPowderML: Use trained machine learning model to predict flowability of virtual powder
TODO VirtualPowderRegression: Use fitted regression model to predict flowability of virtual powder

"""

import pandas
import uuid


class VirtualPowder:

    def __init__(self, composition_map: dict, data: pandas.DataFrame, name: str = None):
        """
        create a new virtual powder given n powders from a dataset with compositions summing to 1.
        This function will
        :param composition_map: a dictionary with format {"sample_id": mass_composition}
        Ex: {"9fc7a768-17b5-4d03-9f78-ef7b2f08ed51": 0.65, "4b799ad8-8ae3-4b6c-afb1-5fed263a5b18" : 0.35}
        :param data: clean labeled data. (do not do particle balancing before generating new virtual powders)
        :param name: name of new sample
        """
        self.data = data
        self.composition_map = composition_map
        self.name = name

    def generate_sample(self) -> pandas.DataFrame:
        """
        generate the virtual powder sample
        :return: virtually generated powder sample
        """
        virtual_powder = self.sample_particles()
        labeled_virtual_powder = self.update_flowability(virtual_powder)
        named_virtual_powder = self.update_name_uuid(labeled_virtual_powder)
        final_virtual_powder = named_virtual_powder.sample(frac=1)  # shuffle order
        return final_virtual_powder

    def sample_particles(self) -> pandas.DataFrame:
        """
        sample the composition powders. this sampling method creates a naive balanced mass of each powder
        :return: sample the raw particle values for the correct composition percentage of each powder
        """
        virtual_powder = None
        for sample_id, mass_composition in self.composition_map.items():
            powder = self.data[self.data['sample_id'] == sample_id]
            sampled_powder = self.sample_a_powder(powder, mass_composition)
            if virtual_powder is None:
                virtual_powder = sampled_powder
            else:
                virtual_powder = virtual_powder.append(sampled_powder)
        return virtual_powder

    @staticmethod
    def sample_a_powder(powder_particles: pandas.DataFrame, mass_composition: float) -> pandas.DataFrame:
        """
        given a powder set of particles and mass composition, sample that percentage of the particles
        :param powder_particles: particles from one powder
        :param mass_composition: float percentage value
        :return: a pand
        """
        count_sample_particles = int(len(powder_particles) * mass_composition)
        sampled_powder = powder_particles.sample(count_sample_particles)
        return sampled_powder

    def update_flowability(self, virtual_powder: pandas.DataFrame):
        """
        get flowability values of composition particles and then
        :param virtual_powder: sampled virtual powders
        :return: virtual powder with updated flowability value
        """
        flow_vals = self.get_flowability_values()
        predicted_powder_flow = self.predict_flowability(flow_vals)
        virtual_powder["flowability"] = predicted_powder_flow
        return virtual_powder

    def get_flowability_values(self) -> dict:
        """
        get the flowability values for a given powder particle
        :return:
        """
        flowability_map = dict()
        for sample_id in self.composition_map.keys():
            flowability = self.data[self.data.sample_id == sample_id].iloc[0].flowability
            flowability_map[sample_id] = flowability
        return flowability_map

    def predict_flowability(self, flowability_values: dict) -> float:
        """
        given the composition percentages and the flowability values of the makeup powders, predict the
        flowabiltiy of the new virtual powder (limited to two particles)
        :param flowability_values:
        :return:
        """
        new_flowability = 0
        for sample_id, flowability in flowability_values.items():
            percentage = self.composition_map[sample_id]
            safe_flow = self.safe_flowability(flowability)
            new_flowability += percentage * safe_flow
        return new_flowability

    @staticmethod
    def inverse_flowability(flowability) -> float:
        """
        given a flowability, convert to inverse flowability. (this function is reversible
        :param flowability: flowability of powder
        :return: return the inverse flowability
        """
        if flowability == 0:
            return 0
        return 1 / flowability

    @staticmethod
    def safe_flowability(flowability, limit=100) -> float:
        """
        account for 0 as infinite flowability
        :param flowability: flowability of powder
        :param limit: large flowability value that all powders will be less than
        :return: return the inverse flowability
        """
        if flowability == 0:
            return limit
        return flowability

    def update_name_uuid(self, virtual_powder: pandas.DataFrame) -> pandas.DataFrame:
        """
        generate the UUID for the sample, in order to include it into dataset
        :param virtual_powder: virtual powder from sampled powders
        :return: virtual powder with updated name and sample_id columns
        """
        sample_id = str(uuid.uuid4())
        if self.name is None:
            self.name = sample_id
        virtual_powder["sample_id"] = sample_id
        virtual_powder["name"] = self.name
        return virtual_powder


# Sample Techniques

class VirtualPowderPercentage(VirtualPowder):
    """
    sample the powders, ignoring the mass equality assumption and create a pure percentage based particle
    mix in virtual powder
    :return: sample the raw particle values for the correct composition percentage of each powder
    """
    def sample_particles(self) -> pandas.DataFrame:
        virtual_powder = None
        # ids_with_lens has structure:
        #       id: [mass_composition, num_particles_of_id]
        ids_with_lens = {}
        min_num_particles = -1
        max_num_particles = -1
        min_particle_proportion = -1
        max_particle_proportion = -1
        min_num_id = None
        max_num_id = None

        # check through our composition map to find the max/min particle counts and proportions
        for sample_id, mass_composition in self.composition_map.items():
            powder = self.data[self.data['sample_id'] == sample_id]

            ids_with_lens[sample_id] = [mass_composition, len(powder)]
            if min_num_particles == -1 or min_num_particles > len(powder):
                min_num_particles = len(powder)
                min_num_id = sample_id

            if max_num_particles == -1 or max_num_particles < len(powder):
                max_num_particles = len(powder)
                max_num_id = sample_id

            if min_particle_proportion == -1 or min_particle_proportion > mass_composition:
                min_particle_proportion = mass_composition

            if max_particle_proportion == -1 or max_particle_proportion < mass_composition:
                max_particle_proportion = mass_composition

        # first, check to see if all the particle counts are equal
        all_equal = True
        id_keys = list(ids_with_lens.keys())
        for i in range(1, len(id_keys)):
            if ids_with_lens[id_keys[i]][1] != ids_with_lens[id_keys[i-1]][1]:
                all_equal = False
                break

        # if each of the powders in our sample are of equal count
        if all_equal:
            for sample_id, mass_composition in self.composition_map.items():
                # sample like normal
                powder = self.data[self.data['sample_id'] == sample_id]
                sampled_powder = self.sample_a_powder(powder, mass_composition)
                if virtual_powder is None:
                    virtual_powder = sampled_powder
                else:
                    virtual_powder = virtual_powder.append(sampled_powder)
        else:
            # calculate the total number of particles we need
            total_particles = int(min_num_particles / min_particle_proportion * max_particle_proportion)

            # grab the min and max powder data
            # NOTE this section isn't usable with >2 particles being mixed!
            min_powder = self.data[self.data['sample_id'] == min_num_id]
            max_powder = self.data[self.data['sample_id'] == max_num_id]

            # check to see if the total number of particles is more or less than the maximum number of
            # particles we could need
            if total_particles > max_num_particles:
                # sample the minimum and maximum powders
                min_sampled_powder = min_powder.sample(int(max_num_particles
                                                       / max_particle_proportion
                                                       * min_particle_proportion))
                max_sampled_powder = max_powder.sample(max_num_particles)
            else:  # if total_particles <= max_num_particles:
                # sample the min and max powders
                min_sampled_powder = min_powder.sample(min_num_particles)
                max_sampled_powder = max_powder.sample(total_particles)

            # append the data to each other
            virtual_powder = min_sampled_powder.append(max_sampled_powder)

        return virtual_powder


class VirtualPowderMass(VirtualPowder):
    def sample_a_powder(self, powder_particles: pandas.DataFrame, mass_composition: float) -> pandas.DataFrame:
        """
        given a powder set of particles and mass composition, sample that percentage of the particles
        :param powder_particles: particles from one powder
        :param mass_composition: float percentage value
        :return: a panda dataframe
        """
        # Calculate the percentage of density this powder should take up
        powder_density_percentage = mass_composition / sum(powder_particles['volume'])

        # We will fill this sampled powder with random particles whose mass all adds up
        # to the specified mass composition
        sampled_powder = None
        # Keeps track of the current mass of the powder
        new_mass_composition = 0

        # Loops through a random ordering of the particles in the powder
        for particle in powder_particles.sample(len(powder_particles)):
            # Calculates the mass of the particle
            calculated_mass = particle['volume'] * powder_density_percentage
            if sampled_powder is None: # Virtual powder is empty
                sampled_powder = particle
            elif calculated_mass + new_mass_composition <= mass_composition:
                # Specified mass not reached yet; add particle to powder, and update current mass
                virtual_powder = virtual_powder.append(sampled_powder)
                new_mass_composition += calculated_mass
        return sampled_powder


# Flowability Prediction


class VirtualPowderML(VirtualPowder):

    def predict_flowability(self, flowability_values: dict) -> float:
        # TODO Implement ML model usage here
        return -1


class VirtualPowderRegression(VirtualPowder):

    def predict_flowability(self, flowability_values: dict) -> float:
        # TODO Implement regression model to predict flowability value - Chris
        return -1