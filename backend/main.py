from typing import Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import io
import numpy as np
from astropy.table import Table
from astropy.io import fits


payload = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # Load DESI zall-pix-fuji.fits file into memory
    zall = Table.read('/data/desi/zall-pix-fuji.fits', format='fits')
    
    # add index
    zall.add_index('TARGETID')

    payload["zall"] = zall
    yield
    # Clean up the payload and release the resources
    payload.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World from FastAPI!"}


def row2dict(row):
    "Convert an astropy table row to a dictionary."
    d = {}
    for key in row.keys():
        value = row[key]
        if isinstance(value, (np.ma.core.MaskedConstant,)):
            d[key] = None
        elif isinstance(value, (np.ndarray,)):
            d[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32, np.int16, np.uint8,)):
            d[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            d[key] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            d[key] = bool(value)
        else:
            d[key] = value
    return d


def rec2dict(rec, ispec):
    "Convert a numpy recarray to a dictionary."
    d = {}
    for key in rec.dtype.names:
        value = rec[key][ispec]
        if isinstance(value, (np.ma.core.MaskedConstant,)):
            d[key] = None
        elif isinstance(value, (np.ndarray,)):
            d[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32, np.int16, np.uint8,)):
            d[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            d[key] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            d[key] = bool(value)
        else:
            d[key] = value
    return d


@app.get("/coadd/{targetid}")
async def get_coadd_spectrum_json(targetid: int, q: Union[str, None] = None):
    
    zall = payload["zall"]

    try:
        r = zall.loc[targetid]

        # Define file path to coadd spectrum
        survey = r['SURVEY']
        program = r['PROGRAM']
        pixnum = r['HEALPIX']
        pix = int(pixnum / 100)
        file_coadd = f"/data/desi/healpix/{survey}/{program}/{pix}/{pixnum}/coadd-{survey}-{program}-{pixnum}.fits"

        # Read index of the spectrum for given targetid
        fibermap = fits.getdata(file_coadd, 'FIBERMAP')
        ispec = np.argwhere(fibermap['TARGETID'] == targetid)[0][0]

        # Read ZALL_PIX_INFO information
        info = row2dict(r)
        
        # Read SCORES information
        scores = fits.getdata(file_coadd, 'SCORES')
        scores = rec2dict(scores, ispec)

        # Read spectra
        data = {}
        with fits.open(file_coadd, memmap=False) as hdul:
            data['b_wavelength'] = hdul['B_WAVELENGTH'].data.tolist()
            data['b_flux'] = hdul['B_FLUX'].section[ispec].tolist()
            data['b_ivar'] = hdul['B_IVAR'].section[ispec].tolist()
            data['b_mask'] = hdul['B_MASK'].section[ispec].tolist()

            data['r_wavelength'] = hdul['R_WAVELENGTH'].data.tolist()
            data['r_flux'] = hdul['R_FLUX'].section[ispec].tolist()
            data['r_ivar'] = hdul['R_IVAR'].section[ispec].tolist()
            data['r_mask'] = hdul['R_MASK'].section[ispec].tolist()

            data['z_wavelength'] = hdul['Z_WAVELENGTH'].data.tolist()
            data['z_flux'] = hdul['Z_FLUX'].section[ispec].tolist()
            data['z_ivar'] = hdul['Z_IVAR'].section[ispec].tolist()
            data['z_mask'] = hdul['Z_MASK'].section[ispec].tolist()

        output = dict(zall_pix_info=info, scores=scores, data=data)

        return output

    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"TARGETID not found in `zall-pix-fuji` table. {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/coadd-csv/{targetid}")
async def get_coadd_spectrum_csv(targetid: int, filter: Union[str, None] = 'R'):
    
    zall = payload["zall"]

    try:
        r = zall.loc[targetid]

        # Define file path to coadd spectrum
        survey = r['SURVEY']
        program = r['PROGRAM']
        pixnum = r['HEALPIX']
        pix = int(pixnum / 100)
        file_coadd = f"/data/desi/healpix/{survey}/{program}/{pix}/{pixnum}/coadd-{survey}-{program}-{pixnum}.fits"

        # Read index of the spectrum for given targetid
        fibermap = fits.getdata(file_coadd, 'FIBERMAP')
        ispec = np.argwhere(fibermap['TARGETID'] == targetid)[0][0]

        # Read spectra
        print(filter)
        with fits.open(file_coadd, memmap=False) as hdul:
            wavelength = hdul[filter+'_WAVELENGTH'].data.astype(float)
            flux = hdul[filter+'_FLUX'].section[ispec]
            ivar = hdul[filter+'_IVAR'].section[ispec]
            mask = hdul[filter+'_MASK'].section[ispec]

        # Create a table
        table = Table([wavelength, flux, ivar, mask], names=('wavelength', 'flux', 'ivar', 'mask'))
        table['wavelength'].info.format = '.1f'
        # table['flux'].info.format = '.5e'
        # table['ivar'].info.format = '.5e'
        
        buffer = io.StringIO()
        table.write(buffer, format='csv')

        # Reset the buffer's position to the start
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="text/csv")

    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")