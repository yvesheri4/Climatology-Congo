#!/bin/bash 

iyear=1985
fyear=2004

indir='/Project/Extraction/Data'
outdir='/Project/Extraction'

# mkdir -p ${outdir}

# select the period from 1931 to 1960
# cdo selyear,${iyear}/${fyear} ${indir}/chirps-v2.0.1985-2004.monthly_p25.nc ${outdir}/chirps_1985_2004.nc

# compute annual mean from multi year monthly data  
# you could also compute annual/seasonal cycle by cdo ymonmean  
cdo yearmonmean ${indir}/chirps.nc ${outdir}/chirps_yearmean.nc
cd ${outdir}
#pwd; ls

# compute standardized anomalies
cdo div -sub chirps_yearmean.nc -timmean chirps_yearmean.nc -timstd chirps_yearmean.nc chirps_yearmean_std.nc

rm -f chirps_yearmean.nc

stations=( Kinshasa Lubumbashi  Goma Mbandaka)
lons=( -4.4419 11.68  1.6585 0.0266)
lats=(  15.266  27.50 29.22  18.297)


for k in $(seq 0 3); do
    # extraction plus proche voisin -- extraction usign the nearest neighbors
    # cdo remapnn,lon=${lons[k]}/lat=${lats[k]} afr_cru_1931_1960_ano_std.nc afr_cru_1931_1960_remapnn_${stations[k]}.nc
    # extraction bilineaire ponderee
    cdo remapdis,lon=${lons[k]}/lat=${lats[k]} chirps_yearmean_std.nc chirps_yearmean_std_remapbil_${stations[k]}.nc

    #ncview afr_cru_1931_1960_remapnn_${stations[k]}.nc afr_cru_1931_1960_remapbil_${stations[k]}.nc

done













