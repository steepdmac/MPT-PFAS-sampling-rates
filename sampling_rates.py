"""
Collection of PFAS specific data and functions to get relevant parameters like diffusion coefficients, partition coefficients,
sampling rates, half-life times to equilibrium, etc.

Starting point for calculations are the following five papers: \
[1] basic theory on sampling rates in PE-tube sampler from Matt Dunn: https://pubs.acs.org/doi/10.1021/acsestwater.2c00384 \
[2] response to [1] from Kees Booij: https://pubs.acs.org/doi/10.1021/acsestwater.3c00085 \
[3] sorption coefficients and Diffusion coefficients of PFAS in high density polyethylen (HDPE): https://www.sciencedirect.com/science/article/pii/S0956053X23006906 \
[4] diffusion coefficients of PFAS in water + temperature correction from Jarod Snook:  https://pubs.acs.org/doi/full/10.1021/acs.est.4c14136 \
[5] diffusion coefficients of PFAS in water (molecular dynamics simulation): https://www.sciencedirect.com/science/article/pii/S037838122300208X \
[6] partition coefficients of PFAS in HLB from Jarod Snook: https://pubs.acs.org/doi/full/10.1021/acsestwater.3c00084
[7] diffusion coefficients of PFAS in water (measured): https://pubs.acs.org/doi/10.1021/acsestwater.4c00631
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional

# define PFAS data class and initialize pfas data

# molecular weights and molecular volumes are calculated solely based on the chemical formula
# Kd values and diffusion coefficients of PFAS in HDPE from [3]
# Kd values of PFAS in HLB from [6]
# diffusion coefficients of PFAS in water from [7]

class PFAS_DATA:

    def __init__(self):

        pfas_data = pd.DataFrame(columns=[
            'Compound', 'molecular weight [g/mol]', 'molecular volume [mL/mol]', 'number of carbons', 'functional group',
            'partition coefficient in HDPE [l/kg]', 'diffusion coefficient in HDPE [cm*cm/s]', 'log of partition coefficient in HLB [mL/g]',
            'STD log of partition coefficient in HLB [mL/g]', 'diffusion coefficient in water at 25C [cm*cm/s]',])
        pfas_data.loc[0, :] = ['TFA', 114.02, 57.79, 2, 'CO2H', np.nan, np.nan, 2.10, np.nan, 9.2e-6]
        pfas_data.loc[1, :] = ['PFPrA', 164.031, 79.42, 3, 'CO2H', np.nan, np.nan, np.nan, np.nan, 6.9e-6]
        pfas_data.loc[2, :] = ['PFBA', 214.04, 101.05, 4, 'CO2H', 5.8, 4.67e-13, 3.25, 0.08, 6.73e-6]
        pfas_data.loc[3, :] = ['PFPeA', 264.05, 122.68, 5, 'CO2H', 5.1, 4.18e-13, 3.81, 0.10, 6.33e-6]
        pfas_data.loc[4, :] = ['PFHxA', 314.05, 144.31, 6, 'CO2H', 6.4, 2.67e-13, 4.40, 0.10, 5.93e-6]
        pfas_data.loc[5, :] = ['PFHpA', 364.05, 165.94, 7, 'CO2H', 5.8, 1.35e-13, 5.23, 0.12, 5.48e-6]
        pfas_data.loc[6, :] = ['PFOA', 414.05, 	187.57, 8, 'CO2H', 9.8, 1.3e-13, 5.97, 0.20, 5.46e-6]
        pfas_data.loc[7, :] = ['PFNA', 464.05, 209.20, 9, 'CO2H', np.nan, np.nan, 6.43, 0.24, 4.75e-6]
        pfas_data.loc[8, :] = ['PFDA', 514.05, 230.83, 10, 'CO2H', 12.5, 5.3e-14, 6.69, 0.27, 4.72e-6]
        pfas_data.loc[9, :] = ['PFUnDA', 564.05, 252.46, 11, 'CO2H', 26.6, np.nan, 6.40, 0.27, 4.51e-6]
        pfas_data.loc[10, :] = ['PFDoDA', 614.10, 274.09, 12, 'CO2H', 62.0, 6.44e-14, 5.89, 0.30, 4.16e-6]
        pfas_data.loc[11, :] = ['PFTrDA', 664.11, 295.72, 13, 'CO2H', 71.9, np.nan, 4.97, 0.18, 3.51e-6]
        pfas_data.loc[12, :] = ['PFTeDA', 714.12, 317.35, 14, 'CO2H', 90.4, 1.04e-14, 4.66, 0.12, 2.08e-6]
        pfas_data.loc[13, :] = ['5:3 FTCA', 342.11, 172.49, 8, 'CO2H', np.nan, np.nan, np.nan, np.nan, 5.53e-6]
        pfas_data.loc[14, :] = ['7:3 FTCA', 442.12, 203.91, 10, 'CO2H', np.nan, np.nan, np.nan, np.nan, 4.61e-6]
        pfas_data.loc[15, :] = ['PFPrS', 250.09, 113.48, 3, 'SO3H', np.nan, np.nan, np.nan, np.nan, np.nan]
        pfas_data.loc[16, :] = ['PFBS', 300.10, 135.11, 4, 'SO3H', 8.3, 2.81e-13, 3.69, 0.02, 6.79e-6]
        pfas_data.loc[17, :] = ['PFPeS', 350.10, 156.74, 5, 'SO3H', 8.8, np.nan, np.nan, np.nan, 6.18e-6]
        pfas_data.loc[18, :] = ['PFHxS', 400.12, 178.37, 6, 'SO3H', 9.8, 2.28e-13, 5.58, 0.10, 5.46e-6]
        pfas_data.loc[20, :] = ['PFHpS', 450.12, 200.00, 7, 'SO3H', 8.2, np.nan, 6.26, 0.18, 5.33e-6]
        pfas_data.loc[21, :] = ['PFOS', 500.12, 221.63, 8, 'SO3H', 12.1, 6.42e-14, 6.16, 0.08, 5.27e-6]
        pfas_data.loc[23, :] = ['PFNS', 550.14, 243.26, 9, 'SO3H', 26.0, np.nan, np.nan, np.nan, 4.80e-6]
        pfas_data.loc[24, :] = ['PFDS', 600.15, 264.89, 10, 'SO3H', 98.3, np.nan, np.nan, np.nan, 4.73e-6]
        pfas_data.loc[25, :] = ['4:2 FTS', 328.15, 163.29, 6, 'SO3H', 3.8, np.nan, 3.91, 0.02, 6.22e-6]
        pfas_data.loc[26, :] = ['6:2 FTS', 428.17, 206.55, 8, 'SO3H', 10.1, np.nan, 5.30, 0.01, 5.54e-6]
        pfas_data.loc[27, :] = ['8:2 FTS', 528.19, 249.81, 10, 'SO3H', 30.8, np.nan, 6.58, 0.30, 4.94e-6]
        pfas_data.loc[28, :] = ['10:2 FTS', 628.20, 293.07, 12, 'SO3H', np.nan, np.nan, np.nan, np.nan, np.nan]
        pfas_data.loc[29, :] = ['FBSA', 299.11, 139.22, 4, 'SO2NH2', np.nan, np.nan, 5.75, 0.13, np.nan]
        pfas_data.loc[30, :] = ['FPeSA', 349.12, 160.85, 5, 'SO2NH2', np.nan, np.nan, np.nan, np.nan, np.nan]
        pfas_data.loc[31, :] = ['FHxSA', 399.13, 182.48, 6, 'SO2NH2', np.nan, np.nan, 7.13, 0.16, np.nan]
        pfas_data.loc[32, :] = ['FHpSA', 449.14, 204.11, 7, 'SO2NH2', np.nan, np.nan, np.nan, np.nan, np.nan]
        pfas_data.loc[33, :] = ['FOSA', 499.14, 225.7, 8, 'SO2NH2', 57.1, np.nan, 6.44, np.nan, 3.48e-6]
        pfas_data.loc[34, :] = ['PFECHS', 462.13, 209.79, 8, 'SO3H', np.nan, np.nan, np.nan, np.nan, np.nan]
        pfas_data.loc[35, :] = ['N-MeFOSAA', 571.21, 275.45, 11, 'N-Me', 62.3, np.nan, np.nan, np.nan, 5.25e-6]
        pfas_data.loc[36, :] = ['N-EtFOSAA', 585.23, 289.54, 12, 'N-Et', 77.4, np.nan, np.nan, np.nan, 5.19e-6]

        pfas_data.index = pfas_data['Compound']
        pfas_data.drop(columns='Compound', inplace=True)

        self.pfas_data = pfas_data

        # parameters from [1]
        self.porosity = 0.35  # proportion
        self.tortuosity = 1.0
        #self.tortuosity = self.porosity/(1 - (1 - self.porosity)**(2/3))  # https://www.sciencedirect.com/science/article/pii/S0012825220304852
        self.delta_tortuosity = 0.1  # estimated
        self.effective_length = 5  # cm
        self.delta_length = 1 # cm
        self.inner_radius = 0.4 # cm
        self.outer_radius = 0.6 # cm
        self.sorbent_weight = 0.6  # g
        self.effective_area = self.outer_radius * 2 * np.pi * self.effective_length  # cm^2
        self.delta_area = self.outer_radius * 2 * np.pi * self.delta_length  # cm^2

    def get_functional_group_mapper(self):
        functional_group_mapper = self.pfas_data['functional group'].to_dict()
        return functional_group_mapper

    def get_number_of_carbons_mapper(self):
        carbon_chain_length_mapper = self.pfas_data['number of carbons'].to_dict()
        return carbon_chain_length_mapper

    # function for the caluclation of Diffusion Coefficient a la Jarod [4]
    def get_diffusion_coefficient_pfas_in_water(self, compound: str, temperature_in_C: float) -> list[float, float, float]:
        """Parses pfas dataframe, calculates diffusion coefficient for pfas in water based on different formluas
        with different temperature corrections and returns relevant parameters.

        :param compound: PFAS compound identifier.
        :type compound: str
        :param temperature_in_C: water temperature in degree celsius
        :type temperature_in_C: float
        :return: [diffusion coefficient of pfas in wate [cm^2/s], partition coefficient of pfas in HDPE[l/kg], diffusion coefficient of pfas in HDPE [cm^2/s]]
        :rtype: list[float, float, float]
        """    

        # function for temperature correction
        D_temperature = lambda D, T:  D * ((273.15 + T) / 298.15) * np.exp((1.37023 * (T - 25) + 0.000836 * (T - 25)**2)/(109 + T))

        # get properties of PFAS compound from data frame
        [M, V, NC, functional_group, Kmw, Dm, Ksw, dKsw, Dw] = self.pfas_data.loc[compound,:]

        # calculate diffusion coefficient in cm*cm/s of PFAS with various laws and average
        # DW_Archie = D_temperature(3.3e-5 * 0.98**2 / M**(1/3), temperature_in_C)
        DW_M = D_temperature(7e-5 * M**(-0.45), temperature_in_C)
        DW_V = D_temperature(1.52e-4 * V**(-0.64), temperature_in_C)
        DW_CN = 1e-2 * (2.1903 - 0.079221 * NC) * np.exp(-1.916e4/(8.31446261815324 * (temperature_in_C + 273.15)))
        return [np.nanmean([DW_V, DW_M, DW_CN]), np.nanstd([DW_V, DW_M, DW_CN]), Kmw, Dm]
    
    def get_log_partition_coefficient_pfas_in_hlb(self, compound: str) -> list[float, float, float]:
        """Parses pfas dataframe, returns partition coefficient of pfas in HLB and its standard deviation."""

        logKsw = self.pfas_data.loc[compound, 'log of partition coefficient in HLB [mL/g]']
        delta_logKsw = self.pfas_data.loc[compound, 'STD log of partition coefficient in HLB [mL/g]']
        if not pd.isna(logKsw) and not pd.isna(delta_logKsw):
            low_Ksw = 10**(logKsw - delta_logKsw)
            Ksw = 10**(logKsw)
            high_Ksw = 10**(logKsw + delta_logKsw)
        elif not pd.isna(logKsw):
            low_Ksw = 10**(logKsw)
            Ksw = 10**(logKsw)
            high_Ksw = 10**(logKsw)
        else:
            low_Ksw = np.nan
            Ksw = np.nan
            high_Ksw = np.nan
        return [low_Ksw, Ksw, high_Ksw]
    
    def plot_diffusion_coefficients(self) -> None:
        D_temperature = lambda D, T:  D * ((273.15 + T) / 298.15) * np.exp((1.37023 * (T - 25) + 0.000836 * (T - 25)**2)/(109 + T))
        for temperature_in_C in [5, 15, 25]:
            fig, ax = plt.subplots()
            for (index, compound) in enumerate(self.pfas_data.index.tolist()):
                [M, V, NC, functional_group, Kmw, Dm, Ksw, dKsw, Dw] = self.pfas_data.loc[compound,:]
                # DW_Archie = D_temperature(3.3e-5 * 0.98**2 / M**(1/3), temperature_in_C)
                DW_M = D_temperature(7e-5 * M**(-0.45), temperature_in_C)
                DW_V = D_temperature(1.52e-4 * V**(-0.64), temperature_in_C)
                DW_CN = 1e-2 * (2.1903 - 0.079221 * NC) * np.exp(-1.916e4/(8.31446261815324 * (temperature_in_C + 273.15)))
                # ax.scatter(x=index, y=DW_Archie, s=5, edgecolors="#0d8631", label='Molecular Mass 1')
                ax.scatter(x=index, y=DW_M, s=5, edgecolors="#7ed11e", label='Molecular Mass')
                ax.scatter(x=index, y=DW_V, s=5, edgecolors="#11E7E4", label='Molecular Volume')
                ax.scatter(x=index, y=DW_CN, s=5, edgecolors="#ce1424", label='Carbon Number')
                ax.errorbar(
                    x=index, y=np.nanmean([DW_V, DW_M, DW_CN]),
                    yerr=np.nanstd([DW_V, DW_M, DW_CN]), xerr=0.3,
                    ecolor="#2043F1", elinewidth=1, label='Model Average'
                    )
                if temperature_in_C==25:
                    ax.scatter(x=index, y=Dw, s=10, edgecolors='k', label='Experimental')
                if index==0:
                    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            ax.set_ylabel('Diffusion coefficient [cm*cm/s]')
            ax.set_xticks([i for i in range(len(self.pfas_data.index.tolist()))])
            ax.set_xticklabels(self.pfas_data.index.tolist())
            ax.tick_params(axis='x', rotation=90)
            ax.grid()
            plt.tight_layout()
            plt.savefig(f'plots/diffusion_coefficients_{temperature_in_C}.png')

    def plot_sampling_rates(self) -> None:
        T=15
        delta_T = 10
        fig, ax = plt.subplots()
        for (index, compound) in enumerate(self.pfas_data.index.tolist()):
            Rs, deltaRs = self.evaluate_sampling_rate(
                compound=compound, temperature_in_C=T, deltaT=delta_T, area_correction=1/5
                )
            ax.errorbar(
                x=index, y=Rs, yerr=deltaRs, xerr=0.3,
                ecolor="#2043F1", elinewidth=1, label='Model Average'
                )
        ax.set_ylabel('Sampling Rate [mL/(day cm)]')
        ax.set_xticks([i for i in range(len(self.pfas_data.index.tolist()))])
        ax.set_xticklabels(self.pfas_data.index.tolist())
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        plt.tight_layout()
        plt.show()
        #plt.savefig(f'plots/diffusion_coefficients_{temperature_in_C}.png')

    def plot_Ksw(self) -> None:
        fig, ax = plt.subplots()
        for (index, compound) in enumerate(self.pfas_data.index.tolist()):
            logKsw = self.pfas_data.loc[compound, 'log of partition coefficient in HLB [mL/g]']
            delta_logKsw = self.pfas_data.loc[compound, 'STD log of partition coefficient in HLB [mL/g]']
            if compound in ['PFPrA', 'TFA']:
                color= 'red'
            else:
                color= 'blue'
            ax.errorbar(
                x=index, y=logKsw, yerr=delta_logKsw, xerr=0.3,
                ecolor=color, elinewidth=1, label='Model Average'
                )
        ax.set_ylabel('log Partition Coefficient in HLB [mL/g]')
        ax.set_xticks([i for i in range(len(self.pfas_data.index.tolist()))])
        ax.set_xticklabels(self.pfas_data.index.tolist())
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        plt.tight_layout()
        plt.show()
        #plt.savefig(f'plots/diffusion_coefficients_{temperature_in_C}.png')

    def plot_half_time_to_equilibrium(self, half: float=0.5) -> None:
        fig, ax = plt.subplots()
        for (index, compound) in enumerate(self.pfas_data.index.tolist()):
            t_half, delta_t_half = self.calculate_half_time_equilibrium(compound=compound, half=half)
            if t_half < 365:
                if compound in ['PFPrA', 'TFA']:
                    color= 'red'
                else:
                    color= 'blue'
                ax.errorbar(
                    x=index, y=t_half, yerr=delta_t_half, xerr=0.3,
                    ecolor=color, elinewidth=1, label='Model Average'
                    )
                ax.text(
                    x=index, y=t_half, s=f"{t_half:.1f}",
                    color=color, fontsize=12, ha='center', va='bottom'
                    )
        ax.set_ylabel(f'Time to {100 * half:.0f} % Equilibrium [days]')
        ax.set_xticks([i for i in range(len(self.pfas_data.index.tolist()))])
        ax.set_xticklabels(self.pfas_data.index.tolist())
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        plt.tight_layout()
        plt.show()
        #plt.savefig(f'plots/diffusion_coefficients_{temperature_in_C}.png')
    
    def evaluate_sampling_rate(
        self, compound: str, temperature_in_C: float, k_WBL: Optional[float] = np.nan, 
        delta_kWBL: Optional[float] = np.nan, area_correction: float = 1, deltaT: float = 0
        ) -> float:

        Dw, DwStd, Kmw, Dm = self.get_diffusion_coefficient_pfas_in_water(compound=compound, temperature_in_C=temperature_in_C)
        if deltaT != 0:
            DwThigh = self.get_diffusion_coefficient_pfas_in_water(
                compound=compound, temperature_in_C=temperature_in_C + deltaT
                )[0]
            DwTlow = self.get_diffusion_coefficient_pfas_in_water(
                compound=compound, temperature_in_C=temperature_in_C - deltaT
                )[0]
            deltaDw = (DwThigh - DwTlow) / 2
        else:
            deltaDw = 0

        # transport equation through membrane
        
        k0 = self.porosity * Dw / (
            self.tortuosity**2 * np.log(self.outer_radius / self.inner_radius) * self.outer_radius
            )
        if not np.isnan(k_WBL):
            k0 = k_WBL * k0 / (k_WBL + k0)
        deltak0 = (self.porosity / (np.log(self.outer_radius / self.inner_radius) * self.outer_radius))  * (
            (DwStd + deltaDw) / (self.tortuosity**2) + 2 * Dw * self.delta_tortuosity / (self.tortuosity**3)
        )
        if not np.isnan(delta_kWBL):
            deltak0 = (1 / (k_WBL + k0)**2) * (k_WBL**2 * deltak0 + k0**2 * delta_kWBL)
        Rs = 8.64e4 * k0 * self.effective_area * area_correction  # calculate sampling rate and convert to L/day
        deltaRs = 8.64e4 * area_correction * (deltak0 * self.effective_area + k0 * self.delta_area)  # calculate uncertainty in sampling rate and convert to L/day

        return [Rs, deltaRs]
    
    def evaluate_sampling_rates_compound_list(
            self, compound_list: list[str], temperature_in_C: float, deltaT: float, area_correction: float=1,
            ) -> list[float]:
        sampling_rates = []
        delta_sampling_rates = []
        for compound in compound_list:
            Rs, deltaRs = self.evaluate_sampling_rate(
                compound=compound, temperature_in_C=temperature_in_C, deltaT=deltaT, area_correction=area_correction
                )
            sampling_rates.append(Rs)
            delta_sampling_rates.append(deltaRs)
        return sampling_rates, delta_sampling_rates
    
    def evaluate_sampling_rates_carbon_number_functional_group(self, compound_list: list[tuple[str, str]], temperature_in_C: float) -> list[float]:
        sampling_rates = []
        delta_sampling_rates = []
        for (functional_group, carbon_chain_length) in compound_list:
            compounds = self.pfas_data.loc[(
                (self.pfas_data['number of carbons'] == carbon_chain_length) &
                (self.pfas_data['functional group'] == functional_group)
                ), :].index.to_list()
            if len(compounds) == 1:
                Rs, deltaRs = self.evaluate_sampling_rate(compound=compounds[0], temperature_in_C=temperature_in_C)
                sampling_rates.append(Rs)
                delta_sampling_rates.append(deltaRs)
            else:
                compound_sampling_rates = []
                compound_delta_sampling_rates = []
                for compound in compounds:
                    Rs, deltaRs = self.evaluate_sampling_rate(compound=compound, temperature_in_C=temperature_in_C)
                    compound_sampling_rates.append(Rs)
                    compound_delta_sampling_rates.append(deltaRs)
                sampling_rates.append(np.nanmean(compound_sampling_rates).astype(float))
                delta_sampling_rates.append(np.nanmean(compound_sampling_rates).astype(float))
        return sampling_rates, delta_sampling_rates
    
    def calculate_half_time_equilibrium(self, compound: str, half: float = 0.5) -> list[float, float]:
        low_Ksw, Ksw, high_Ksw = self.get_log_partition_coefficient_pfas_in_hlb(compound=compound)
        Rs, deltaRs = self.evaluate_sampling_rate(compound=compound, temperature_in_C=15, deltaT=5)
        t_half = np.log(1/(1-half)) * self.sorbent_weight * Ksw / Rs # in days
        delta_t_half = np.log(1/(1-half)) * self.sorbent_weight * (
            ((high_Ksw - low_Ksw)) / (2 * Rs) + deltaRs * Ksw / (Rs**2)
            )
        return t_half, delta_t_half
    
    def determine_kWBL_from_sampling_rate(
            self, compound: str, temperature_in_C: float, Rs: float, area_correction: float=1
            ) -> float:
        """Determine kWBL from sampling rate and diffusion coefficient in water.

        :param compound: PFAS compound identifier.
        :type compound: str
        :param temperature_in_C: water temperature in degree celsius
        :type temperature_in_C: float
        :return: kWBL [cm/s]
        :rtype: float
        """
        
        Dw, _, _, _ = self.get_diffusion_coefficient_pfas_in_water(compound=compound, temperature_in_C=temperature_in_C)
        kWBL_theoretical =  self.porosity * Dw / (
            self.porosity * Dw * self.effective_area * 8.64e4 * area_correction / Rs - \
            np.log(self.outer_radius / self.inner_radius) * self.outer_radius * self.tortuosity**2
        )

        if kWBL_theoretical < 0 or kWBL_theoretical > 1e-4:
            kWBL_theoretical = np.nan

        return kWBL_theoretical

if __name__ == '__main__':
    pfas_data_class = PFAS_DATA()
    pfas_data_class.plot_diffusion_coefficients()
    # pfas_data_class.plot_sampling_rates()
    # pfas_data_class.plot_Ksw()
    # pfas_data_class.plot_half_time_to_equilibrium(half=0.5)
    # t_half, delta_t_half = pfas_data_class.calculate_half_time_equilibrium(compound='PFBS')
    # print(t_half, delta_t_half)