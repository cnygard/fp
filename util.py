import torch

def LED_angles(positions: torch.Tensor, config: dict):
    # TODO: these should be parameters in the config
    H = 90.88
    h = 1
    n = 1.52
    startX, startY = 20, 18
    angles = torch.clone(positions)
    for led in range(positions.shape[0]):
        x0, y0 = positions[led, 0], positions[led, 1]
        l = torch.sqrt(x0**2+y0**2) # distance of LED from origin

        xoff = 0 # initial guess where beam enters bottom of slide
        thetag = -torch.asin(l/torch.sqrt(l**2+H**2)/n) # get angle of beam in glass from Snell's law
        xint = h*torch.tan(thetag) # find where the beam exits the top of the slide
        xoff = xoff-xint # modify guess where beam enters bottom of slide by this amount

        # repeat the above procedure until the beam exits the top of the slide
        # within 1 micron of center
        while torch.abs(xint) > .001:
            thetag = -torch.asin((l-xoff)/torch.sqrt((l-xoff)**2+H**2)/n)
            xint = xoff+h*torch.tan(thetag)
            xoff = xoff-xint

        # angle under the glass and angle over the cover slip
        # FPM treats this as the angle in the sample so assumes the sample has
        # refractive index = 1.0
        theta = torch.asin((l-xoff)/torch.sqrt((l-xoff)**2+H**2))

        NAt = torch.abs(torch.sin(theta));
        posX = positions[led, 0] - startX
        posY = positions[led, 1] - startY
        distX = posX * config['LED_spacing']
        tanX = distX / config['object_distance']
        distY = posY * config['LED_spacing']
        tanY = distY / config['object_distance']
        tans = torch.tensor([tanX, tanY])
        rads = torch.atan(tans)
        degs = torch.rad2deg(rads)
        # thetal = torch.atan2(y0,x0) # angle of LED in x-y plane
        # kx  = -NAt*torch.cos(thetal);
        # ky  = -NAt*torch.sin(thetal);
        angles[led, 0] = -NAt*degs[0] 
        angles[led, 1] = -NAt*degs[1] 
    return angles
