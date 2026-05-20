from astropy.io import fits


def print_fits_header(filename):
    """Open a FITS file and print each HDU header."""
    with fits.open(filename) as hdul:
        for i, hdu in enumerate(hdul):
            name = hdu.name if hdu.name else "PRIMARY"
            print(f"--- HDU {i}: {name} ---")
            print(hdu.header)
            print()


if __name__ == "__main__":
    fits_path = r"Z:\20251015\iLocater_lab_20251015_0003.fits"
    print_fits_header(fits_path)
