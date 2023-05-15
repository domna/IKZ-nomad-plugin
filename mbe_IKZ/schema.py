import numpy as np
import re
from datetime import datetime as dt
import pandas as pd
import json

from nomad.units import ureg
from nomad.metainfo import (
    MSection, Package, Quantity, SubSection, MEnum, Reference, Datetime, Section)
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.eln import PublicationReference
from nomad.datamodel.metainfo.workflow2 import Link
from nomad.datamodel.metainfo.eln import Entity, Activity, SampleID
from nomad.datamodel.util import parse_path


m_package = Package(name='mbe_IKZ')


def create_archive(entry_dict, context, file_name):
    if not context.raw_path_exists(file_name):
        with context.raw_file(file_name, 'w') as outfile:
            json.dump(entry_dict, outfile)
        context.process_updated_raw_file(file_name)


class GrowthRecipeStep(EntryData):
    '''
    The datafile.asl is parsed into a repeated section of the eln.
    '''
    epi_step = Quantity(
        type=np.dtype(np.int64),
        description='Sequential number for the steps in the recipe')

    name = Quantity(
        type=str,
        description='Name of the current recipe step',
        a_eln=dict(component='StringEditQuantity'))

    nesting_level = Quantity(
        type=np.dtype(np.int64),
        description='Consecutive number for repeated steps in superlattice applications')

    periods = Quantity(
        type=np.dtype(np.int64),
        description='Number of repetitions for one step in superlattice applications')

    thickness = Quantity(
        type=np.dtype(np.float64),
        description='Set growth thickness')

    elapsed_time = Quantity(
        type=np.dtype(np.float64),
        unit='minute',
        description='Elapsed time in minutes')

    T_substrate = Quantity(
        type=str,  # np.dtype(np.float64),
        # unit='celsius',
        description='Temperature of the substrate, >> and << indicate a temperature ramp')

    rotation = Quantity(
        type=np.dtype(np.float64),
        unit='rpm',
        description='Rotation of the substrate holder. + clockwise and - counterclockwise')

    Si_evap = Quantity(
        type=np.dtype(np.float64),
        description='Input power of Silicon source in percent with 100% = max. Power in combination with a check during growth with mass spectrometer')

    Ge_hts = Quantity(
        type=np.dtype(np.float64),
        description='High temperature source for Germanium growth')

    Ga = Quantity(
        type=np.dtype(np.float64),
        description='Gallium source')

    In = Quantity(
        type=np.dtype(np.float64),
        description='Indium source')

    susi = Quantity(
        type=np.dtype(np.float64),
        description='Defect')

    e_evap2 = Quantity(
        type=np.dtype(np.float64),
        description='Electron beam evaporator for Germanium')

    Si_evap = Quantity(
        type=np.dtype(np.float64),
        description='Input power of silicon source in percent with 100% = max. Power')

    obs_m = Quantity(
        type=np.dtype(np.float64),
        description='Oxygen source')


class GrowthLogStep(EntryData):
    '''
    The datafile.daa is parsed into a repeated section of the eln.
    '''
    name = Quantity(
        type=str,
        description='Name of the current logged step')
    timestamp = Quantity(
        type=Datetime,
        description='The timestamp corresponding to the start of logging',
        a_eln=dict(component='DateTimeEditQuantity'))
    elapsed_time = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
        description='Time for each EpiStep in h:mm:ss.ms/NaN/At Run Time, automatic calculated time which is needed for each step based on calibration data, can be altered manually')
    setp = Quantity(
        type=np.dtype(np.int64),
        shape=['*'],
        unit="celsius",
        a_eln=dict(defaultDisplayUnit='celsius'),
        description='Setpoint which is set either manually or from a recipe')
    # setpcomp = Quantity(
    #     type=np.dtype(np.int64),
    #     shape=['*'],
    #     unit="celsius",
    #     a_eln=dict(defaultDisplayUnit='celsius'),
    #     description='ignore this quantity')
    procv = Quantity(
        type=np.dtype(np.int64),
        shape=['*'],
        unit="celsius",
        a_eln=dict(defaultDisplayUnit='celsius'),
        description='The actual present value relative to the process loop')
    # procvcomp = Quantity(
    #     type=np.dtype(np.int64),
    #     shape=['*'],
    #     unit="celsius",
    #     a_eln=dict(defaultDisplayUnit='celsius'),
    #     description='ignore this quantity')
    ysetp = Quantity(
        type=np.dtype(np.int64),
        shape=['*'],
        unit="celsius",
        a_eln=dict(defaultDisplayUnit='celsius'),
        description='Target value; it can be seen as the Y-axis in a time vs. P-Loop graph')
    yprocv = Quantity(
        type=np.dtype(np.int64),
        shape=['*'],
        unit="celsius",
        a_eln=dict(defaultDisplayUnit='celsius'),
        description='The value which has to be set in order to achieve the Procv value. i.e. this value is calibrated to get e.g. the right temp. to the substrate')
    power = Quantity(
        type=np.dtype(np.int64),
        shape=['*'],
        description='The actual current power in percent')
    # accumulatedprocv = Quantity(
    #     type=np.dtype(np.int64),
    #     shape=['*'],
    #     unit="celsius / second",
    #     a_eln=dict(defaultDisplayUnit='celsius / second'),
    #     description='ignore this quantity')
    switchcontrol = Quantity(
        type=bool,
        shape=['*'],
        description='The target switch state of the shutter, (0 = Off  1 = On)')
    switchmonitor = Quantity(
        type=bool,
        shape=['*'],
        description='The actual state of the shutter, 0 = Off  1 = On')
    commsstatus = Quantity(
        type=bool,
        shape=['*'],
        description='The status of the whole process, 0 = failed  1 = OK')


class GrowthRecipe(EntryData, Activity):
    '''
    A task for MBE synthesis.
    '''
    data_file = Quantity(
        type=str,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    name = Quantity(
        type=str,
        description='Name of the recipe',
        a_eln=dict(component='StringEditQuantity'))

    timestamp = Quantity(
        type=Datetime,
        description='Timestamp parsed from the recipe file',
        a_eln=dict(component='DateTimeEditQuantity'))

    tasks = SubSection(section_def=GrowthRecipeStep, repeats=True)

    def normalize(self, archive, logger):
        if self.data_file:
            logger.info('found datafile.asl')
            with archive.m_context.raw_file(self.data_file, 'r',
                                            encoding='unicode_escape') as file:

                line = file.readlines()[0:1]
                name = re.search('NAME:(.+?)RUN:', line[0])
                if name:
                    self.name = name.group(1)
                start_date = re.search('(?<=RUN:)(.*)', line[0])
                if start_date:
                    regex = re.compile("[0-9]{1}[a-z]{2}")
                    cardinal_day = re.findall(regex, start_date[0])[0]
                    ordinal_day = cardinal_day[:-2]
                    clean = re.sub(cardinal_day, ordinal_day, start_date[0])
                    self.timestamp = dt.strptime(str(clean).replace("-", "").strip(),
                                                 '%H:%M:%S %A %d %B %Y')
                data = pd.read_csv(file.name, encoding='unicode_escape',
                                   skiprows=[0, 1, 2, 3], sep="\t")
                growth_recipes: dict = {}
                elapsed = pd.Timedelta(seconds=0, microseconds=0, minutes=0, hours=0)
                for step, value in enumerate(data[' Time']):
                    for key in data.keys():
                        if str(data[key][step]) == 'nan':
                            data[key][step] = int(0)
                    growth_recipes[step] = GrowthRecipeStep()
                    setattr(growth_recipes[step], 'epi_step',
                            data['EpiStep'][step])
                    setattr(growth_recipes[step], 'name',
                            data[' Type'][step])
                    setattr(growth_recipes[step], 'nesting_level',
                            data[' Nesting Level'][step])
                    setattr(growth_recipes[step], 'periods',
                            data[' Periods'][step])
                    thickness = f"{data[' Thickness'][step]}".replace('-', '0').replace('nm', '')
                    setattr(growth_recipes[step], 'thickness', float(thickness))
                    if (value == "At Run Time" or isinstance(value, (float, int))):
                        time = pd.Timedelta(seconds=0, microseconds=0, minutes=0, hours=0)
                    else:
                        clean = str(data[' Time'][step]).replace('(t=', '').replace(')', '')
                        (time, usec) = clean.split('.')
                        microsec = '{:<06}'.format(usec)
                        (hour, min, sec) = time.split(':')
                        time = pd.Timedelta(seconds=int(sec),
                                            microseconds=int(microsec),
                                            minutes=int(min),
                                            hours=int(hour))
                    elapsed += time
                    setattr(growth_recipes[step], 'elapsed_time', elapsed.value / 60000000000)
                    setattr(growth_recipes[step], 'T_substrate', str(data[' Tsub(°C)'][step]))
                    setattr(growth_recipes[step], 'rotation',
                            np.float64(str(data[' Rotation(rpm)'][step]).replace(' ', '')))
                    setattr(growth_recipes[step], 'Si_evap',
                            np.float64(data['Si_evap(%)'][step]))
                    setattr(growth_recipes[step], 'Ge_hts',
                            np.float64(data['Ge_HTS(%)'][step]))
                    setattr(growth_recipes[step], 'Ga', np.float64(data['Ga(%)'][step]))
                    setattr(growth_recipes[step], 'In', np.float64(data['In(%)'][step]))
                    setattr(growth_recipes[step], 'susi',
                            np.float64(data['SUSI 63(%)'][step]))
                    setattr(growth_recipes[step], 'e_evap2',
                            np.float64(data['e-evap2(%)'][step]))
                    setattr(growth_recipes[step], 'Si_evap',
                            np.float64(data['Si-evap(%)'][step]))
                    setattr(growth_recipes[step], 'obs_m', np.float64(data['OBS_M(%)'][step]))
                    # self.measurements.sensor_102 = data['102'].to_numpy() * ureg('celsius')
                self.tasks = []
                for recipe_step in growth_recipes.values():
                    self.tasks.append(recipe_step)


class GrowthLog(EntryData, Activity):
    '''
    A task for MBE synthesis.
    '''
    data_file = Quantity(
        type=str,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    name = Quantity(
        type=str,
        description='FILL THE DESCRIPTION',
        a_eln=dict(component='StringEditQuantity'))

    timestamp = Quantity(
        type=Datetime,
        description='FILL THE DESCRIPTION',
        a_eln=dict(component='DateTimeEditQuantity'))

    tasks = SubSection(section_def=GrowthLogStep, repeats=True)

    def normalize(self, archive, logger):
        if self.data_file:
            logger.info('found datafile.daa')
            self.name = 'Recipe Log file'
            with archive.m_context.raw_file(self.data_file, 'r', encoding='unicode_escape') as file:
                filelines = file.readlines()
                steps = []
                self.tasks = []
                for index, line in enumerate(filelines):
                    if f"H\t" in line:
                        steps.append(index)
                steps.append(len(filelines))
                for index, step in enumerate(steps):
                    if index < (len(steps) - 1):
                        step_obj = GrowthLogStep()
                        step_lines = steps[index + 1] - step
                        assert('L\t' in filelines[steps[index] + 1])
                        setattr(step_obj, 'name', filelines[steps[index]].split('\t')[1])
                        data = pd.read_csv(file.name,
                                           encoding='unicode_escape',
                                           encoding_errors='ignore',
                                           skiprows=[i for i in range(step + 1)] + [step + 2],
                                           nrows=step_lines - 3,
                                           sep="\t")
                        setattr(step_obj, 'setp', data['Setp'].to_numpy())
                        # setattr(step_obj, 'setpcomp', data['SetpComp'].to_numpy())
                        setattr(step_obj, 'procv', data['Procv'].to_numpy())
                        # setattr(step_obj, 'procvcomp', data['ProcvComp'].to_numpy())
                        setattr(step_obj, 'ysetp', data['YSetp'].to_numpy())
                        setattr(step_obj, 'yprocv', data['YProcv'].to_numpy())
                        setattr(step_obj, 'power', data['Power'].to_numpy())
                        # setattr(step_obj, 'accumulatedprocv', data['AccumulatedProcv'].to_numpy())
                        timesteps = []
                        switch_controls = []
                        switch_monitors = []
                        comms_status = []
                        start_time = dt.strptime(f"{data['Time'][0].strip()} "
                                                 f"{data['Date'][0]}".strip(),
                                                 '%H:%M:%S.%f %Y-%m-%d')
                        setattr(step_obj, 'timestamp', start_time)
                        for index in range(step_lines - 3):
                            current = dt.strptime(f"{data['Time'][index].strip()} {data['Date'][index]}".strip(),
                                                  '%H:%M:%S.%f %Y-%m-%d')
                            timesteps.append(pd.Timedelta.total_seconds(current - start_time))
                            switch_controls.append(bool(data['SwitchControl'][index]))
                            switch_monitors.append(bool(data['SwitchMonitor'][index]))
                            comms_status.append(bool(data['CommsStatus'][index]))
                        setattr(step_obj, 'elapsed_time', timesteps)
                        setattr(step_obj, 'switchcontrol', switch_controls)
                        setattr(step_obj, 'switchmonitor', switch_monitors)
                        setattr(step_obj, 'commsstatus', comms_status)
                        self.tasks.append(step_obj)


class Steps(MSection):
    '''Class autogenerated from yaml schema.'''
    m_def = Section(
        a_eln={
            "properties": {
                "order": [
                    "duration",
                    "ratio",
                    "step_number"]}}
                    )
    step_number = Quantity(
        type=int,
        description='sequential number of the step on going',
        a_eln={
            "component": "NumberEditQuantity"})

    ratio = Quantity(type=str, a_eln={"component": "StringEditQuantity"})

    duration = Quantity(
        type=np.float64,
        description='Duration of the current step in seconds',
        unit='second',
        a_eln={
            "component": "NumberEditQuantity",
            "defaultDisplayUnit": "second"})


class SubstratePreparation(EntryData, Activity):
    '''Class autogenerated from yaml schema.'''

    method = Quantity(
        type=str,
        default="Substrate Preparation")

    steps = SubSection(section_def=Steps, repeats=True)


class SampleCut(EntryData, Activity):
    ''' An Activity that can be used for cutting a sample in multiple ones. '''

    number_of_samples = Quantity(
        type=int,
        description='The number of samples generated from this "Sample Cut" Task.',
        a_eln=dict(component='NumberEditQuantity'))
    input_sample = SubSection(sub_section=Link, repeats=True, description=(
        'All the links to sections that represent the inputs for this task.'))
    output_samples = SubSection(sub_section=Link, repeats=True, description=(
        'All the links to sections that represent the outputs for this task.'))

    def normalize(self, archive, logger):
        super(SampleCut, self).normalize(archive, logger)

        if self.inputs:
            if len(self.inputs) != 1:
                logger.warn(f"Error in '{self.name}': Only one input expected, but {len(self.inputs)} inputs given.")
            if self.output_samples:
                logger.warn(f"Error in '{self.name}': No output samples expected,"
                            f" but {len(self.output_samples)} output samples given.")
            if not self.number_of_samples:
                logger.warn(f"Error in '{self.name}': 'number_of_samples' expected, but None found.")
            if not (SampleID in attr for attr in self.inputs[0].section.m_proxy_type.resolve(self.inputs[0].section).__dict__.values()):
                logger.warn(f"Error in '{self.inputs[0].name}': 'SampleID' class expected, but None found.")
            for attribute in self.inputs[0].section.m_proxy_type.resolve(self.inputs[0].section).__dict__.values():
                if isinstance(attribute, SampleID):
                    parent_attribute = attribute
                    if not parent_attribute.sample_short_name:
                        logger.warn(f"Error in '{self.inputs[0].name}': 'sample_short_name' expected, but None found.")

            _, upload_id, mainfile, _, _ = parse_path(self.inputs[0].section.m_proxy_value)
            if '.data.archive.yaml' in mainfile:
                pass
            else:
                parent = self.m_context.load_archive(mainfile, upload_id, None)
                mainfile = parent.metadata.entry_name

            parent_object: Section = self.inputs[0].section
            collection = CollectionOfSystems()
            from nomad.datamodel import EntryArchive, EntryMetadata
            for sample_index in range(self.number_of_samples):
                children_name = f"{mainfile.split('.data.archive.yaml')[0]}_{sample_index}"
                children_object = parent_object.m_copy(deep=True)
                for attribute in children_object.__dict__.values():
                    if isinstance(attribute, SampleID):
                        attribute.sample_short_name = f'{parent_attribute.sample_short_name}_{sample_index}'
                        attribute.sample_id = None
                children_object.lab_id = None
                # children_object.results.eln.lab_ids = []
                filename = f"{children_name}.data.archive.yaml"
                create_archive(EntryArchive(data=children_object).m_to_dict(), self.m_context, filename)
                collection.systems.append(Link(section=f"../upload/raw/{filename}#data",
                                          name=children_name))

            collection_filename = f"{mainfile.split('.data.archive.yaml')[0]}_collection.data.archive.yaml"
            create_archive(EntryArchive(metadata=EntryMetadata(entry_type=CollectionOfSystems),
                           data=collection).m_to_dict(), self.m_context, collection_filename)
            self.output_samples = []
            self.output_samples.append(Link(section=f"../upload/raw/{collection_filename}#data",
                                            name=collection_filename.split('.data.archive.yaml')[0]))


class CalibrationDateSources(EntryData, Activity):
    '''Class autogenerated from yaml schema.'''

    source_material = Quantity(
        type=str,
        description='FILL',
        a_eln={
            "component": "StringEditQuantity"})

    calibration_date = Quantity(
        type=Datetime,
        description='FILL',
        a_eln={
            "component": "DateTimeEditQuantity"})


class CollectionOfSystems(Entity, EntryData):
    '''
    A base class for a batch of materials. Each component of the batch is
    of a (sub)type of `System`.
    '''

    systems = SubSection(sub_section=Link, repeats=True, description=(
        'All the links to sections that represent the members of this batch.'))


class MbeExperiment(EntryData):
    """MBE experiment"""

    m_def = Section(
        a_template={
            "publication_reference": {},
            "substrate_preparation": {},
            "substrate_cut": {},
            "growth_recipe": {},
            "calibration_date_sources": {},
            "growth_log": {}}
    )

    publication_reference = SubSection(section_def=PublicationReference) #, repeats=True)
    substrate_preparation = SubSection(section_def=SubstratePreparation) #, repeats=True)
    substrate_cut = SubSection(section_def=SampleCut) #, repeats=True)
    growth_recipe = SubSection(section_def=GrowthRecipe) #, repeats=True)
    calibration_date_sources = SubSection(section_def=CalibrationDateSources) #, repeats=True)
    growth_log = SubSection(section_def=GrowthLog) #, repeats=True)


m_package.__init_metainfo__()
