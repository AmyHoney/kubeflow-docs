.. _install-tkgs:

======================================
Install Kubeflow on vSphere with Tanzu
======================================

This document will guide you to install Kubeflow on vSphere with Tanzu.

.. note::
	In this guide, we will install Kubeflow 1.6.1. Configurations may be slightly different for other versions.

Prerequisites
=============

Adhere to the following requirements before deploying Kubeflow package on TKGS clusters.

For the deployment on vSphere with Tanzu (TKGS), Kubeflow will be installed on a Tanzu Kubernetes Cluster (TKC). So before the deployment of Kubeflow, you should get vSphere and TKC ready.

- For a greenfield deployment (no vSphere with Tanzu deployed on servers yet), you will need to deploy vSphere with Tanzu first. Please refer to VMware official document `vSphere with Tanzu Configuration and Management <https://docs.vmware.com/en/VMware-vSphere/7.0/vmware-vsphere-with-tanzu/GUID-152BE7D2-E227-4DAA-B527-557B564D9718.html>`__.

- With the proper vSphere with Tanzu installation and configuration, you will need to provison TKGS clusters. See `Workflow for Provisioning Tanzu Kubernetes Clusters Using the TKGS v1alpha2 API <https://docs.vmware.com/en/VMware-vSphere/7.0/vmware-vsphere-with-tanzu/GUID-3040E41B-8A54-4D23-8796-A123E7CAE3BA.html>`__.

- If you want to use GPU resources on vSphere platform, setup vGPU TKG with document `Deploy AI/ML Workloads on Tanzu Kubernetes Clusters <https://docs.vmware.com/en/VMware-vSphere/7.0/vmware-vsphere-with-tanzu/GUID-2B4CAE86-BAF4-4411-ABB1-D5F2E9EF0A3D.html>`__.

- Connect to the cluster from your client host. See `Connect to a Tanzu Kubernetes Cluster as a vCenter Single Sign-On User <https://docs.vmware.com/en/VMware-vSphere/7.0/vmware-vsphere-with-tanzu/GUID-AA3CA6DC-D4EE-47C3-94D9-53D680E43B60.html>`__

- Install ``kctrl``, a kapp-controller's native CLI on your client host. It will be used to install Kubeflow Carvel Package. See `Installing kapp-controller CLI: kctrl <https://carvel.dev/kapp-controller/docs/v0.40.0/install/#installing-kapp-controller-cli-kctrl>`__

Deploy Kubeflow package on TKGS clusters
========================================

Step 1: Add package repository
------------------------------

.. code-block:: shell

	kubectl create ns carvel-kubeflow-namespace
	kubectl config set-context --current --namespace=carvel-kubeflow-namespace

	kctrl package repository add \
	  --repository kubeflow-carvel-repo \
	  --url projects.registry.vmware.com/kubeflow/kubeflow-carvel-repo:1.6.1

Step 2: Create ``config.yaml`` file
-----------------------------------

Create a ``config.yaml`` file which would be used in later Kubeflow installation.

.. note::
	This YAML file would be created based on values schema of Kubeflow package, i.e. the configurations. More details can be found in :ref:`values schema table`.

.. code-block:: shell

    cat > config.yaml << 'EOF'

    service_type: "LoadBalancer"

    IP_address: ""

    CD_REGISTRATION_FLOW: True

    EOF

Step 3: Install Kubeflow package
--------------------------------

.. code-block:: shell
  
  kctrl package install \
      --wait-check-interval 5s \
      --wait-timeout 30m0s \
      --package-install kubeflow \
      --package kubeflow.community.tanzu.vmware.com \
      --version 1.6.1 \
      --values-file config.yaml

This may take a few minutes, so please wait patiently. You should see a "Succeed" message in the end.

    .. image:: ../_static/install-tkgs-deploySucceed.png

Step 4: Access Kubeflow
-----------------------

Now, we are ready to access our deployed Kubeflow in browser and start our journey.

To access Kubeflow, we need to get the IP address. There are three options.

- When you set ``service_type`` to ``LoadBalancer``, run the command below and visit ``EXTERNAL-IP`` of ``istio-ingressgateway``.

  .. code-block:: shell

      kubectl get svc istio-ingressgateway -n istio-system

      # example output:
      # NAME                   TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)                                                                      AGE
      # istio-ingressgateway   LoadBalancer   198.51.217.125   10.105.151.142   15021:31063/TCP,80:30926/TCP,443:31275/TCP,31400:30518/TCP,15443:31204/TCP   11d
      
      # In this example, we should visit http://10.105.151.142:80
- When you set ``service_type`` to ``NodePort``, run the command below and visit ``nodeIP:nodePort``.

  .. code-block:: shell

      kubectl get svc istio-ingressgateway -n istio-system

      # example output:
      # NAME                   TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)                                                                      AGE
      # istio-ingressgateway   NodePort   198.51.217.125   <none>        15021:31063/TCP,80:30926/TCP,443:31275/TCP,31400:30518/TCP,15443:31204/TCP   11d

      kubectl get nodes -o wide

      # example output:
      # NAME                                                      STATUS   ROLES                  AGE   VERSION            INTERNAL-IP     EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION      CONTAINER-RUNTIME
      # v1a2-v1-23-8-tkc-v100-8c-dcpvc-4zct9                      Ready    control-plane,master   26d   v1.23.8+vmware.2   10.105.151.73   <none>        Ubuntu 20.04.4 LTS   5.4.0-124-generic   containerd://1.6.6
      # v1a2-v1-23-8-tkc-v100-8c-workers-zwfx4-77b7df85f7-f7f6f   Ready    <none>                 26d   v1.23.8+vmware.2   10.105.151.74   <none>        Ubuntu 20.04.4 LTS   5.4.0-124-generic   containerd://1.6.6
      # v1a2-v1-23-8-tkc-v100-8c-workers-zwfx4-77b7df85f7-l5mp5   Ready    <none>                 26d   v1.23.8+vmware.2   10.105.151.75   <none>        Ubuntu 20.04.4 LTS   5.4.0-124-generic   containerd://1.6.6

      ## In this example, any one of the following will work:
      # http://10.105.151.73:30926
      # http://10.105.151.74:30926
      # http://10.105.151.75:30926
- Use ``port-forward``. Then visit the IP address of your client-side machine.

  .. code-block:: shell

      kubectl port-forward -n istio-system svc/istio-ingressgateway --address 0.0.0.0 8080:80

      # if you are running the command locally, you should visit http://localhost:8080

You can then use the IP to access Kubeflow in browser.

    .. image:: ../_static/install-tkgs-login.png

For the first time you login after deployment, you would be guided to namespace creation page after login.

    .. image:: ../_static/install-tkgs-createNS.png

You should then see the Kubeflow home page.

    .. image:: ../_static/install-tkgs-home.png

Troubleshooting
===============

More ``kctrl`` commands can be found in `kapp-controller's native CLI documentation <https://carvel.dev/kapp-controller/docs/v0.43.2/management-command/>`__.

Delete the Kubeflow package
---------------------------

To uninstall the Kubeflow package:

   .. code-block:: shell

      kctrl package installed delete --package-install kubeflow

When deleting the Kubeflow package, some resources may get stuck at ``deleting`` status. To solve this problem:

   .. code-block:: shell

      # take namespace knative-serving as an example
      kubectl patch ns knative-serving -p '{"spec":{"finalizers":null}}'
      kubectl delete ns knative-serving --grace-period=0 --force

Reconciliation issue
--------------------

Kapp-controller keeps reconciliating Kubeflow, which prevents you from editing a Kubeflow resource. In this case, you may want to pause or trigger the reconciliation of Kubeflow.


- To pause the reconciliation of a package installation:

   .. code-block:: shell

      kctrl package installed pause --package-install kubeflow

- To trigger the reconciliation of a package installation:

   .. code-block:: shell

      kctrl package installed kick --package-install kubeflow --wait --wait-check-interval 5s --wait-timeout 30m0s

Inspect package installation
----------------------------

- To check the status of package installation:

   .. code-block:: shell

      kubectl get PackageInstall kubeflow -o yaml

- To print the status of App created by package installation:

   .. code-block:: shell

     kctrl package installed status --package-install kubeflow

Update package configurations
-----------------------------

To update the configuration of Kubeflow package using an updated configuration file (i.e., ``config.yaml``):

.. code-block:: shell

    kctrl package installed update --package-install kubeflow --values-file config.yaml

.. _values schema table:

Values schema
-------------

To inspect values schema (configurations) of the Kubeflow package, run following command:

.. code-block:: shell

	kctrl package available get -p kubeflow.community.tanzu.vmware.com/1.6.1 --values-schema

We summarize some important values schema in below table.

==================================== ======================================================================= ======== ==============================================================================================================================================
Key 	  							 Default 																 Type     Description
==================================== ======================================================================= ======== ==============================================================================================================================================
CD_REGISTRATION_FLOW                 true                                                                    boolean  Turn on Registration Flow, so that Kubeflow Central Dashboard will prompt new users to create a namespace (profile)
IP_address                           ""                                                                      string   EXTERNAL_IP address of istio-ingressgateway, valid only if service_type is LoadBalancer  
service_type                         LoadBalancer                                                            string   Service type of istio-ingressgateway. Available options: "LoadBalancer" or "NodePort"
==================================== ======================================================================= ======== ==============================================================================================================================================

.. seealso::

    `Get started with Charmed Kubeflow <https://charmed-kubeflow.io/docs/get-started-with-charmed-kubeflow>`_
