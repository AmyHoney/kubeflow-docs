========
Overview
========

This section introduces how to install and configure vSphere Enterprise Kubeflow. vSphere Enterprise Kubeflow can run on various types of setups, including physical vSphere cluster, virtual hosts, and public cloud. Step-by-step guide will be provided here, with trouble shooting suggestions to some commonly seen installation failures.

Integrating vSphere Enterprise Kubeflow with authentication services and Gitlab enhances the platform's security and collaboration capabilities. For authentication, vSphere Enterprise Kubeflow can be integrated with existing Identity Providers like LDAP, Active Directory, or OAuth2-compliant services. Additionally, vSphere Enterprise Kubeflow can be connected to Gitlab for version control and collaboration on machine learning projects. This integration enables users to synchronize their code, pipelines, and notebooks with Gitlab repositories, streamlining the development process and fostering a collaborative environment.


.. note::
   Note that some sections might be optional, depending on your environment.

KFE supports the following Kubernetes versions:

.. tabs::

   .. tab:: vSphere

      vSphere 8.0

   .. tab:: Alibaba Cloud

      Ubuntu 20.04

   .. tab:: Nimbus

      Ubuntu 20.04
