
"""
Two commonly-used tuple structures in our API are the __particle__ and the __decay product__:

```
my_particle = (name, p_x, p_y, traj_x, traj_y)
my_product = (name, p_x, p_y)
```

Here `name` is a string, one of: `'proton'`, `'pion'`, `'muon'`, `'electron'`, or `'neutrino'`.  `p_x` and `p_y` are the momentum components of a particle in some reference frame.  `traj_x` and `traj_y` are arrays of distances, corresponding to the history of a particle's position.

(Why not just use "particle" tuples for everything?  A design decision to avoid potential confusion.  "Product" tuples will be created _outside_ the lab frame, and they won't have any position assigned to them until we do some work.)


### Special relativity:
* `lorentz_beta_gamma(beta_x, beta_y)`: Computes and returns the dimensionless Lorentz factors `(beta, gamma)` from the components of velocity.
* `boost_momentum(m, px, py, beta_x, beta_y)`: Computes and returns `(px_boost, py_boost)`, the components of momentum in the new frame of reference after a boost by $(\beta_x, \beta_y)$.
* `momentum_to_velocity(m, px, py)`: Returns the dimensionless velocity (`beta_x`, `beta_y`) corresponding to the given momentum and mass.
* `velocity_to_momentum(m, beta_x, beta_y)`: Returns the momentum vector components (`p_x`, `p_y`) corresponding to the given dimensionless velocity and mass.  Not used in the main loop, but useful for initializing particles with a given speed.

### Particle interactions:
* `check_for_interaction(particle, dt, h)`: Checks to see if a particle will interact within time `dt` at height `h` above the Earth's surface (so $h=0$ is at the surface, which we take to be average sea level.)  Returns `True` if a decay or downscatter occurs, and `False` otherwise.
* `particle_decay(name)`: Simulates the decay of a pion or muon in its rest frame.  Returns a tuple `(product_Y, product_Z)` containing _decay product_ tuples for each decay product.
* `proton_downscatter(beta_x , beta_y)`: Simulates the downscattering of a proton in its rest frame.  The inputs are the current components of relativistic velocity of the proton, which are used to determine the momentum of the recoiling nitrogen-14 nucleus in the proton rest frame.  Returns a tuple `(product_pi_1, product_pi_2, product_p)` containing _decay product_ tuples for each downscattering product.

### Other functions:
* `boost_product_to_lab(product, beta_x, beta_y)`: Given a decay product tuple `(name, px, py)` (see above) and the two components of velocity `(beta_x, beta_y)` of the parent particle, returns the tuple `(name, p_x_lab, p_y_lab)` giving the product's momentum in the lab frame, i.e. after boosting the momentum by $(-\beta_x, -\beta_y)$ (_note the minus signs since we're boosting __to the lab frame__.)_
* `particle_mass(name)`: Given a particle name, return its mass (in whatever units you are using.)


### Testing:
* `run_API_tests()`: A custom function that should use assertions to test the other functions implemented in the API.  If all tests are passed, the function should simply return the `None` object.  You should implement __at least six (6) tests__ inside this function on your own.

## Main loop

The below code implements the "main loop", running the Monte Carlo given an initial state, timestep, and number of timesteps to evolve.  Once you have implemented the API above, add this code to your `cosmic.py` module, then call it to run the Monte Carlo simulation!

"""

# Intiating Code Setting up physical constants
import numpy as np
import math

## Important constants
particle_names = ['proton', 'electron', 'pion', 'muon', 'neutrino']

## Unit conversions
km_per_ls = 2.997e5

# Computes beta and gamma from components.
def lorentz_beta_gamma(beta_x, beta_y):
    '''
    Calculates lorentz transform coefficients for particles

      Args:
          - beta_x, beta_y
            - in our unit system, this is just the velocity components

      Returns:
          - (beta, gamma): relatavistic coefficients
    '''
    # Compute beta
    #gamma is a scalar, so I have to take the magnitude of the beta vector
    beta = np.sqrt((beta_x)**2 + (beta_y)**2)

    # Verify beta at least 1

    if beta >= 1:
        raise ValueError("Beta cannot be >= 1!")

    # Compute gamma
    gamma = 1/np.sqrt(1-beta**2)


    return (beta, gamma)

# Compute new momentum components, following a boost to a new frame given by (beta_x, beta_y)
# Note: you might need to treat the massless case separately!
def boost_momentum(m, px, py, beta_x, beta_y):
    # Note care unit choices can be helfpul (hint: cancel out "c")! # do this by setting c=1?
    # Figure out E beta, gamma
    '''
    uses relatavistic coefficients to boost to a different reference frame


    we know gamma  and beta from beta_x and beta_y - we can implement lorentz_beta_gamma

    from openstax university physics volume 3: E^2 = (pc)^2 +(mc^2)^2
      - once again, energy is a scalar, thus, we will have to take the magnitude of the total momentum,
      this takes all velocity directions into account
      -the same value of E will be used in both px_boost and py_boost


  Args:
      - m: mass of particle, in MeV/c^2
      - (px, py): momentum of particle, in MeV/c
      -(beta_x, beta_y): velocity vector of particle # beta_x and beta_y are equal to velcity because c = 1 in our unit system!

  Returns:
      - (px_boost, py_boost): momentum in new reference frame
    '''
    #apply relatavistic energy to particles using their mass and momentum
    beta, gamma = lorentz_beta_gamma(beta_x, beta_y)
    E = np.sqrt((px**2 + py**2) + (m)**2) # remember what units we're using, c=1 for particle physics units, this is why we just have m**2 and p**2

    px_boost= px - (gamma * E * beta_x)+(gamma-1)*beta_x*((beta_x*px + beta_y*py)/beta**2)

    py_boost= py - (gamma * E * beta_y)+(gamma-1)*beta_y*((beta_x*px + beta_y*py)/beta**2)


    return (px_boost, py_boost)

# Given a particle with mass m, convert momentum vector to velocity vector
def momentum_to_velocity(m, px, py): #is this function supposed to give us the beta values for the functions above?
  # this is still part of our boosting process
  #we're trying to find the speed of the particles so we can map their trajectories? - this is useful for a presentation
  """
  Given a particle with mass m, convert momentum vector
    to velocity vector.

  Args:
      - m: mass of particle, in MeV/c^2
      - (px, py): momentum of particle, in MeV/c

  Returns:
      - (beta_x, beta_y): velocity vector of particle # beta_x and beta_y are equal to velcity because c = 1 in our unit system!
  """
    #Massive particle
  if m>0:
    p = np.sqrt(px**2 + py**2)
    beta_x = px / (m * np.sqrt(1 + (p/m)**2)) #remember, c = 1
    beta_y = py / (m * np.sqrt(1 + (p/m)**2))

      # Massless case is special!
  elif m == 0:
    #the speed is equal to c in the same direction of p vector
    #we need to calculate the angle of p vector relative to the horizontal of the vector
    #I asked claude if there was a numpy library to calculate the angle... I forgot about arctan2
    momentum_angle = np.arctan2(py, px)
    #vx = cos(theta)/c , but c = 1, so we'll say vx = cos)(theta).  same logic applies to y
    beta_x = np.cos(momentum_angle)
    beta_y = np.sin(momentum_angle)

  return (beta_x, beta_y)

# Opposite of momentum_to_velocity: given mass and velocity, compute the momentum vector
def velocity_to_momentum(m, beta_x, beta_y):
  '''
  converts particle velocity to momentum

  Args:
      - m: mass of particle, in MeV/c^2
      - (beta_x, beta_y): momentum of particle, in MeV/c

  Returns:
      - (beta_x, beta_y): velocity vector of particle # beta_x and beta_y are equal to velcity because c = 1 in our unit system!
  '''
  v = np.sqrt(beta_x**2 + beta_y**2)
  px = (m*beta_x)/(np.sqrt(1-v**2))
  py = (m*beta_y)/(np.sqrt(1-v**2))

  return (px, py)

# Test to see if a particle will decay/scatter within timestep dt
def check_for_interaction(particle, dt, h):

    '''
    Uses random monte carlo process to test if a particle will interact/decay to produce a cosmic ray shower

    Args:
        - particle: tuple with ('name', px, py, traj_x, traj_y)
        - dt: length of timestep
        -starting height

    Returns:
        -True or False
        - np.random.rand() < p
          -depending on random variable, returns if the probability of decay is above a threshold

    '''

    #begin by unpacking particle for useful info
    name = particle[0]
    p_x = particle[1]
    p_y = particle[2]
    mass = particle_mass(name)

    #use p_x and p_y to calculate beta and gamma using momentum_to_velocity
    beta_vector = momentum_to_velocity(mass, p_x, p_y)

    beta_magnitude = np.sqrt(beta_vector[0]**2 + beta_vector[1]**2) # use this to calculate gamma(this is the same as velocity magnitude!)

    gamma = 1/np.sqrt(1 - (beta_magnitude)**2)


    ## Time constants from cosmic_assign
    tau_p = np.exp(h/7) * (1/beta_magnitude)*1.14*(10**-5)# s
    tau_pi = gamma * 2.603*(10**-8)# s
    tau_mu = gamma * 2.197*(10**-6)  # s

    proton_energy = np.sqrt((p_x**2 + p_y**2) + (mass)**2)

    # Consider cases for different particles
    # Compute probability of interaction
    if name == 'proton' and proton_energy >= 1218:
      p = 1 - np.exp(-dt / tau_p)
    #check for proton threshold energy
    elif name == 'proton' and proton_energy < 1218: #API idea: test this edge case!!!
      return False

    elif name == 'pion':
      p = 1 - np.exp(-dt / tau_pi)
    elif name == 'muon':
      p = 1 - np.exp(-dt / tau_mu)
    else:
      return False



    # Interaction happens with probability p
    return np.random.rand() < p

# Simulate the decay X --> YZ of a pion or muon. Here, Y is a muon (electron), and Z is a neutrino
def particle_decay(name):
    '''
    -performs particle decay of pions or muons- decay only occurs once
    -draws from random theta to determine direction of decay\

    Args:
        - 'name': name of particle

    Returns:
        -(particle_Y, particle_Z) : decay product tuples of either muon or pion

    '''
    # consider pion and muon cases
    theta = np.random.uniform(0, 2*np.pi)
    if name == 'pion':
      productY_name = 'muon'
      Y_total_momentum = (particle_mass('pion')**2 - particle_mass('muon')**2) / (2*particle_mass('pion'))#originally, I was missing parenthesis and it was giving me problems.  Claude helped me recognize this
      Ypx = float(Y_total_momentum * np.cos(theta))
      Ypy = float(Y_total_momentum * np.sin(theta))

    elif name == 'muon':
      productY_name = 'electron'
      Y_total_momentum = (particle_mass('muon')**2 - particle_mass('electron')**2) / (2*particle_mass('muon'))
      Ypx = float(Y_total_momentum * np.cos(theta))
      Ypy = float(Y_total_momentum * np.sin(theta))

    particle_Y = (productY_name, Ypx, Ypy)

    particle_Z = ('neutrino', -Ypx, -Ypy)

    return (particle_Y, particle_Z)

# Simulate the downscattering of a proton off of an air molecule, p --> p + pi + pi
def proton_downscatter(beta_x, beta_y):
    '''
    -performs simulation of proton downscattering in the atmosphere


    Args:
        -  beta_x, beta_y: proton speed

    Returns:
        -(particle_pi_1, particle_pi_2, particle_p) : decay product tuples of proton interaction

    '''


    #creating the particle tuple for the two pion proudcts
    b = 1/500

    #pion1
    y1 = np.random.uniform()
    theta1 = np.random.uniform(0, 2*np.pi)
    Pion1_Energy = particle_mass('pion') - (1/b)*np.log(1-y1)
    pion1_Momentum = np.sqrt((Pion1_Energy**2)-(particle_mass('pion')**2))

    pion1_px = pion1_Momentum * np.cos(theta1)
    pion1_py = pion1_Momentum * np.sin(theta1)

    pion1_momentum_vector = (pion1_px, pion1_py)

    particle_pi_1 = ('pion', pion1_px, pion1_py)


    #pion2
    y2 = np.random.uniform()
    theta2 = np.random.uniform(0, 2*np.pi)
    Pion2_Energy = particle_mass('pion') - (1/b)*np.log(1-y2)
    pion2_Momentum = np.sqrt((Pion2_Energy**2)-(particle_mass('pion')**2))

    pion2_px = pion2_Momentum * np.cos(theta2)
    pion2_py = pion2_Momentum * np.sin(theta2)

    pion2_momentum_vector = (pion2_px, pion2_py)

    particle_pi_2 = ('pion', pion2_px, pion2_py)

    #proton product
    N14_mass = 13040
    # In the proton's rest frame, the N-14 nucleus approaches with velocity (-beta_x, -beta_y)
    N14_initial_momentum = np.array(velocity_to_momentum(N14_mass, -beta_x, -beta_y)) # Corrected to use -beta_x, -beta_y, gemini suggested this change, as well as the conversion to an np array
    N14_initial_momentum_magnitude = np.linalg.norm(N14_initial_momentum)
    N14_px0 = N14_initial_momentum[0]
    N14_py0 = N14_initial_momentum[1]

    N14_initial_energy = np.sqrt((N14_px0**2 + N14_py0**2) + (N14_mass)**2)

    N14_E_prime = N14_initial_energy - Pion1_Energy - Pion2_Energy


    N14_post_collision_momentum_magnitude = np.sqrt(N14_E_prime**2 - N14_mass**2)

    delta_Pn_prime = N14_initial_momentum * (1 - (N14_post_collision_momentum_magnitude / N14_initial_momentum_magnitude)) # This is a vector

    #use delta_pn_prime to find proton momentum vector

    proton_momentum = delta_Pn_prime - np.array(pion1_momentum_vector) - np.array(pion2_momentum_vector)

    proton_px = proton_momentum[0]
    proton_py = proton_momentum[1]

    particle_p = ('proton', proton_px, proton_py)



    return (particle_pi_1, particle_pi_2, particle_p)

# Given a decay product tuple (name, px, py)
# and the two components of velocity (beta_x, beta_y) of the parent particle,
# returns the tuple (name, px_lab, py_lab) giving the product's momentum
# in the lab frame, i.e. after boosting the momentum by (-beta_x, -beta_y)
# (note the minus signs since we're boosting to the lab frame.)
def boost_product_to_lab(product, beta_x, beta_y):

    '''
    -boosts interaction products to lab reference frame for observation


    Args:
        -product: product particle tuple
        -  beta_x, beta_y: proton speed

    Returns:
        -(product_name, px_lab, py_lab) : product name and momentum components observed in lab

    '''
    product_name = product[0]
    m = particle_mass(product_name)
    px = product[1]
    py = product[2]


    lab_momentum = boost_momentum(m, px, py, -beta_x, -beta_y)

    px_lab = lab_momentum[0]
    py_lab = lab_momentum[1]


    return (product_name, px_lab, py_lab)

# Given a particle name, return its mass (in whatever units you are using.)
def particle_mass(name):
  '''
    -returns mass for a given particle


    Args:
        -particle name

    Returns:
        -particle mass

    '''
  mass = 0
  if name == 'electron':
    mass = 0.5110
  elif name == 'proton':
    mass = 938.3
  elif name == 'pion':
    mass = 139.6
  elif name == 'muon':
    mass = 105.7
  elif name == 'neutrino':
    mass = 0
  return mass

"""## API Tests of your design
Be sure throughout ... or at the end here at the end to have your __Six (6)__ additional API tests _you write_ (tests are provided and are correct and useful)
"""

# run_API_tests() to run API Tests
## Test various modules (especally in compbination)
## with the Main Loop below you can test various features


#TEST 1
#testing to see if momentum_to_velocity and velocity_to_momentum circle back to the original value-Claude gave me this idea but I implemented it
#starting with velocity and circling back with momentum
def run_API_tests():
    beta_x_test = 0.5
    beta_y_test = 0.6
    test_particle = 'muon'
    test_mass = particle_mass(test_particle)

    px_test, py_test = velocity_to_momentum(test_mass, beta_x_test, beta_y_test)

    beta_x_result, beta_y_result = momentum_to_velocity(test_mass, px_test, py_test)

    assert np.abs(beta_x_test-beta_x_result) <= 1e-6 #claude suggested the absolute value
    assert np.abs(beta_y_test-beta_y_result) <= 1e-6



    #starting with momentum and circling back with velocity
    px_test = 29
    py_test = 20
    test_particle = 'muon'
    test_mass = particle_mass(test_particle)

    BetaX_test, BetaY_test = momentum_to_velocity(test_mass, px_test, py_test)

    p_x_result, p_y_result = velocity_to_momentum(test_mass, BetaX_test, BetaY_test)
    assert np.abs(px_test-p_x_result) <= 1e-6 #claude suggested the absolute value
    assert np.abs(py_test-p_y_result) <= 1e-6


    #TEST 2
    #Testing if particle_decay conserves momentum

    #testing the pion case:

    decay_products_pion = particle_decay('pion')

    test_momentum_magnitude = np.sqrt(decay_products_pion[0][1]**2 + decay_products_pion[0][2]**2)
    calculated_momentum_magnitude = 29.78391834
    assert np.abs(test_momentum_magnitude-calculated_momentum_magnitude) <= 1e-4

    #testing the muon case:
    decay_products_muon = particle_decay('muon')

    test_momentum_magnitude = np.sqrt(decay_products_muon[0][1]**2 + decay_products_muon[0][2]**2)
    calculated_momentum_magnitude = 52.8487648
    assert np.abs(test_momentum_magnitude-calculated_momentum_magnitude) <= 1e-4



    #TEST 3
    #testing the example usage in MAIN LOOP
    #we're observing a muon, so an electron and neutrino should decay
    #testing correct decay products
    init_p_x, init_p_y = velocity_to_momentum(particle_mass('muon'), 0.85, -0.24) # beta ~ 0.8, relativistic
    init_particles = [ ('muon', init_p_x, init_p_y, np.array([0]), np.array([1e-4]) ) ]  # ~ 30 km height in light-sec
    particles = run_cosmic_MC(init_particles, 1e-5, 1000, attenuation = False)

    names = [particle[0] for particle in particles]#originally this was a for loop, claude suggested list comprehension for code optimization

    assert 'electron' in names
    assert 'neutrino' in names


    #TEST 4
    #making sure all particles are below the speed of light
    #test to make sure beta<1
    #use the same initial conditions as used in test 3
    for particle in particles:
      name = particle[0]
      if name == 'neutrino':#claude added this continue line since neutrinos are massless and have beta = c
        continue
      px = particle[1]
      py = particle[2]
      beta_x, beta_y = momentum_to_velocity(particle_mass(name), px, py)
      beta = lorentz_beta_gamma(beta_x, beta_y)[0]

      assert beta < 1



      #TEST 5
      #testing the massless neutrino, we should get beta = 1

      beta_x, beta_y = momentum_to_velocity(particle_mass(neutrino), 50, 50)
      beta = np.sqrt(beta_x**2 + beta_y**2)
      assert np.abs(beta-1)<= 1e-4




      #TEST 6
      #testing to make sure particle_mass returns correct mass

      assert np.abs(particle_mass(proton)-938.3) <= 1e-6
      assert np.abs(particle_mass(electron)-0.5110) <= 1e-6
      assert np.abs(particle_mass(pion)-139.6) <= 1e-6
      assert np.abs(particle_mass(muon)-105.7) <= 1e-6
      assert np.abs(particle_mass(neutrino)-0) <= 1e-6

## MAIN LOOP
## You can just copy this code below into your cosmic.py
## but review it so you undestand what it is calling

def run_cosmic_MC(particles, dt, Nsteps, attenuation=True):
    """
    Main loop for cosmic ray Monte Carlo project.

    Arguments:
    =====
    * particles: a list of any length, containing "particle" tuples.
        A particle tuple has the form:
            (name, p_x, p_y, traj_x, traj_y)
        where p_x and p_y are momentum vector components, and
        traj_x and traj_y are NumPy arrays of distance.

    * dt: time between Monte Carlo steps, in seconds.
    * Nsteps: Number of steps to run the Monte Carlo before returning.
    * attenuation: if True, protons and muons with too little energy are pruned
      from the simulation.  (This is a crude simulation of the real-world effect
      that charged particles moving through the air will lose energy.)

    Returns:
    =====
    A list of particle tuples.

    Example usage:
    =====

    >> init_p_x, init_p_y = velocity_to_momentum(particle_mass('muon'), 0.85, -0.24) # beta ~ 0.8, relativistic
    >> init_particles = [ ('muon', init_p_x, init_p_y, np.array([0]), np.array([1e-4]) ) ]  # ~ 30 km height in light-sec
    >> particles = run_cosmic_MC(init_particles, 1e-5, 100)

    The 'particles' variable after running should contain three particle tuples: the
    initial muon, an electron, and a neutrino.  (Even though the muon decays,
    we keep it in the final particle list for its trajectory.)

    """

    E_threshold = 2000 # MeV - change this constant if your unit system is different!!

    stopped_particles = []
    for step_i in range(Nsteps):
        updated_particles = []

        for particle in particles:
            # Unpack particle tuple
            (name, p_x, p_y, traj_x, traj_y) = particle
            (beta_x, beta_y) = momentum_to_velocity(particle_mass(name), p_x, p_y)

            # Check for interaction
            does_interact = check_for_interaction(particle, dt, traj_y[-1])

            if does_interact:
                if name == 'proton':
                    decay_products = proton_downscatter(beta_x, beta_y)
                else:
                    stopped_particles.append(particle)
                    decay_products = particle_decay(name)

                # Transform products back to lab frame
                for product in decay_products:
                    (product_name, product_p_x, product_p_y) = boost_product_to_lab(product, beta_x, beta_y)

                    ## Clean up - attentuation of particles with too little energy
                    if (attenuation):
                        E_product = np.sqrt(product_p_x**2 + product_p_y**2 + particle_mass(product_name)**2)
                        if E_product < E_threshold:
                            if product_name == 'proton':
                                # Avoid losing proton tracks, since downscatter doesn't add stopped particles
                                stopped_particles.append(particle)
                            if product_name != 'neutrino':
                                # Skip decay product, don't append to updated_particles
                                continue

                    # If this was a proton scatter, then the "new" proton is
                    # the same as the original, so keep track of its trajectory!
                    if name == 'proton' and product_name == 'proton':
                        product_traj_x = traj_x
                        product_traj_y = traj_y
                    else:
                        product_traj_x = np.array([traj_x[-1]])
                        product_traj_y = np.array([traj_y[-1]])

                    # Make new particle tuple and append
                    product_particle = (product_name, product_p_x, product_p_y,
                                        product_traj_x, product_traj_y)
                    updated_particles.append( product_particle )
            else:
                # Doesn't interact, so compute motion
                traj_x = np.append(traj_x, traj_x[-1] + beta_x * dt)
                traj_y = np.append(traj_y, traj_y[-1] + beta_y * dt)

                updated_particles.append( (name, p_x, p_y, traj_x, traj_y) )


        # Run next timestep
        particles = updated_particles

    # Add stopped particles back to list and return
    particles.extend(stopped_particles)
    return particles