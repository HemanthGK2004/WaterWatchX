import React from 'react';
import { Link } from 'react-router-dom';
import { Droplet, Heart, Users, Shield, ArrowRight } from 'react-feather';

const initiatives = [

  {
    title: 'Water Quality Monitoring',
    description: 'Deploying IoT-based sensors to monitor water quality in real-time, ensuring safe water for urban and rural areas.',
    image: 'https://images.unsplash.com/photo-1600585154340-be6161a56a0c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    impact: '50+ monitoring stations installed',
    icon: <Shield className="text-teal-500" size={24} />
  },
 
  {
    title: 'Water Quality Monitoring',
    description: 'Deploying IoT-based sensors to monitor water quality in real-time, ensuring safe water for urban and rural areas.',
    image: 'https://images.unsplash.com/photo-1600585154340-be6161a56a0c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    impact: '50+ monitoring stations installed',
    icon: <Shield className="text-teal-500" size={24} />
  },
  {
    title: 'Community Water Education',
    description: 'Educating communities on water conservation and hygiene practices to reduce waterborne diseases.',
    image: 'https://images.unsplash.com/photo-1593113598332-cd288d649433?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    impact: '5,000+ participants trained',
    icon: <Users className="text-purple-500" size={24} />
  },
];

const HealthInitiatives = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-blue-50">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-r from-blue-600 to-teal-600 text-white py-24 overflow-hidden">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1470114716159-e389f8712fda?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80')] bg-cover bg-center opacity-20"></div>
        <div className="container mx-auto px-4 text-center relative z-10">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
            Transforming Lives Through <span className="text-blue-200">Clean Water</span>
          </h1>
          <p className="text-xl md:text-2xl max-w-3xl mx-auto mb-8">
            Our innovative water initiatives bring sustainable solutions to communities in need.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              to="/donate"
              className="flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-full font-medium hover:bg-blue-50 transition-all shadow-lg hover:shadow-xl"
            >
              Support Our Work <Heart size={18} />
            </Link>
            <Link
              to="/initiatives"
              className="flex items-center gap-2 bg-transparent border-2 border-white text-white px-6 py-3 rounded-full font-medium hover:bg-white hover:bg-opacity-10 transition-all"
            >
              Learn More <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-blue-50 p-8 rounded-xl text-center">
              <Droplet className="mx-auto text-blue-600" size={32} />
              <h3 className="text-4xl font-bold text-gray-800 mt-4">10,000+</h3>
              <p className="text-gray-600 mt-2">Families Served</p>
            </div>
            <div className="bg-teal-50 p-8 rounded-xl text-center">
              <Shield className="mx-auto text-teal-600" size={32} />
              <h3 className="text-4xl font-bold text-gray-800 mt-4">50+</h3>
              <p className="text-gray-600 mt-2">Monitoring Stations</p>
            </div>
            <div className="bg-purple-50 p-8 rounded-xl text-center">
              <Users className="mx-auto text-purple-600" size={32} />
              <h3 className="text-4xl font-bold text-gray-800 mt-4">5,000+</h3>
              <p className="text-gray-600 mt-2">People Trained</p>
            </div>
          </div>
        </div>
      </section>

      {/* Initiatives Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <span className="inline-block bg-blue-100 text-blue-600 px-4 py-1 rounded-full text-sm font-medium mb-4">
              Our Initiatives
            </span>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4">
              Creating Lasting <span className="text-blue-600">Water Solutions</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              We implement sustainable water programs that address immediate needs while building long-term resilience.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {initiatives.map((initiative, index) => (
              <div
                key={index}
                className="group bg-white rounded-xl shadow-lg overflow-hidden transition-all duration-300 hover:-translate-y-2 hover:shadow-xl"
              >
                <div className="relative h-60 overflow-hidden">
                  <img
                    src={initiative.image}
                    alt={initiative.title}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                  <div className="absolute top-4 right-4 bg-white p-3 rounded-full shadow-md">
                    {initiative.icon}
                  </div>
                </div>
                <div className="p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-3">{initiative.title}</h3>
                  <p className="text-gray-600 mb-4">{initiative.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center gap-1 text-blue-600 font-medium">
                      <Heart size={16} />
                      {initiative.impact}
                    </span>
                    <Link
                      to="/initiatives/details"
                      className="flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium transition-colors"
                    >
                      Learn more <ArrowRight size={16} />
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="relative py-24 bg-[url('https://images.unsplash.com/photo-1508193638397-1c4234db14d8?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80')] bg-cover bg-center bg-fixed">
        <div className="absolute inset-0 bg-blue-800/80"></div>
        <div className="container mx-auto px-4 text-center relative z-10">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
            Join Our Movement for <span className="text-blue-200">Clean Water</span>
          </h2>
          <p className="text-xl text-blue-100 max-w-2xl mx-auto mb-8">
            Together, we can ensure every community has access to safe, clean water.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              to="/volunteer"
              className="flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-full font-medium hover:bg-blue-50 transition-all shadow-lg hover:shadow-xl"
            >
              Volunteer With Us <Users size={18} />
            </Link>
            <Link
              to="/contact"
              className="flex items-center gap-2 bg-transparent border-2 border-white text-white px-6 py-3 rounded-full font-medium hover:bg-white hover:bg-opacity-10 transition-all"
            >
              Contact Our Team <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HealthInitiatives;