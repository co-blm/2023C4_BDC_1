# coding=utf-8
from sklearn.metrics import roc_auc_score
import numpy as np

# 24
fault_types=['NO_FAULT', '3d82f4ad7f114cbdbd469fc897b001a1', '05be78908e3c4818b1caa00b71d8bb11', '6cc7dc7bb5fa4327a20c883ab00ab2fe', '8b18231981e0440488bbac370b1464cf', 
             '2170e75abdf54178afcd5ffecb387eee', '4287f5cca47742008a8fb965908e5dea', '00c1ba198361424c9597328ea33d0d15', 'c453a975de5148e4b1c47be258a646c9', '15b7f6577fec4c16b01ee2e053b1f201', 
             'ea4bdf00441c4157a99a9c72bb7f4eb2', '14eb4630112b4ce9bd88d93104b4570e', '53f1acb37db941b8b9c77dfefecb157b', '8b3eee3cc4fe4568b5ba4125c1a4047f', 'e97a387ed0204878b0660f0090bfacd6', 
             'f1023ca9976e4a5eaaaaed244acd2f4a', '122ec12af3744773b9b04c6c8e929711', 'faf90b12d1cf478e810172eb6aced658', '0d1304f1f40743dea03be55bca96c32b', '03d1f58da52d49dbb815cda9be061d25', 
             '36c4ac32f7504f13b7aef941de9ecc81', 'node-worker1', 'node-worker2', 'node-worker3']

# 113
metric_tags= ['node_cpu_seconds_total', 'node_memory_Active_file_bytes', 'node_load1', 'node_disk_reads_merged_total', 'container_network_transmit_errors_total', 
              'node_network_transmit_queue_length', 'node_memory_Slab_bytes', 'node_disk_io_time_weighted_seconds_total', 'node_network_transmit_errs_total', 'node_disk_info', 
              'node_memory_Inactive_anon_bytes', 'node_disk_reads_completed_total', 'node_memory_DirectMap4k_bytes', 'node_memory_SwapFree_bytes', 'cpm', 
              'node_memory_Inactive_file_bytes', 'container_network_receive_packets_dropped_total', 'node_memory_Inactive_bytes', 'node_network_mtu_bytes', 'node_network_protocol_type', 
              'container_cpu_system_seconds_total', 'container_network_transmit_bytes_total', 'container_network_transmit_packets_dropped_total', 'node_network_address_assign_type', 'node_network_carrier_changes_total', 
              'node_memory_HugePages_Rsvd', 'node_network_net_dev_group', 'error_count', 'node_memory_Mlocked_bytes', 'node_memory_Active_bytes', 
              'node_network_device_id', 'node_memory_Unevictable_bytes', 'container_network_receive_bytes_total', 'node_memory_Shmem_bytes', 'node_memory_MemAvailable_bytes', 
              'node_memory_VmallocChunk_bytes', 'container_cpu_cfs_throttled_seconds_total', 'node_network_flags', 'node_memory_Mapped_bytes', 'node_memory_SReclaimable_bytes', 
              'node_disk_write_time_seconds_total', 'node_network_iface_id', 'node_memory_Buffers_bytes', 'node_network_iface_link_mode', 'node_disk_read_bytes_total', 
              'node_memory_Committed_AS_bytes', 'node_network_transmit_carrier_total', 'node_memory_Dirty_bytes', 'node_load15', 'node_network_transmit_compressed_total', 
              'node_memory_SwapCached_bytes', 'node_network_speed_bytes', 'container_network_transmit_packets_total', 'node_network_receive_multicast_total', 'node_network_info', 
              'node_memory_Cached_bytes', 'node_memory_PageTables_bytes', 'node_disk_writes_completed_total', 'node_network_receive_packets_total', 'node_disk_written_bytes_total', 
              'container_network_receive_errors_total', 'node_cpu_guest_seconds_total', 'node_memory_SUnreclaim_bytes', 'node_memory_MemTotal_bytes', 'node_disk_io_time_seconds_total', 
              'node_memory_CommitLimit_bytes', 'node_network_transmit_colls_total', 'node_network_receive_frame_total', 'node_memory_KernelStack_bytes', 'node_memory_Hugepagesize_bytes', 
              'node_memory_HugePages_Total', 'node_memory_Writeback_bytes', 'node_network_transmit_packets_total', 'success_rate', 'node_memory_VmallocUsed_bytes', 
              'node_network_receive_drop_total', 'container_network_receive_packets_total', 'node_disk_read_time_seconds_total', 'node_network_transmit_drop_total', 'node_disk_writes_merged_total', 
              'node_network_receive_fifo_total', 'container_cpu_cfs_throttled_periods_total', 'container_cpu_user_seconds_total', 'node_network_dormant', 'node_network_transmit_bytes_total', 
              'node_load5', 'node_network_receive_compressed_total', 'node_memory_HugePages_Free', 'node_memory_MemFree_bytes', 'node_memory_CmaTotal_bytes', 
              'node_memory_NFS_Unstable_bytes', 'node_memory_AnonHugePages_bytes', 'node_network_receive_errs_total', 'container_cpu_cfs_periods_total', 'node_memory_Bounce_bytes', 
              'node_memory_AnonPages_bytes', 'node_memory_VmallocTotal_bytes', 'node_memory_SwapTotal_bytes', 'container_cpu_usage_seconds_total', 'node_memory_HugePages_Surp', 
              'node_network_up', 'node_memory_CmaFree_bytes', 'node_network_carrier', 'node_memory_Active_anon_bytes', 'node_network_receive_bytes_total', 
              'resp_time', 'container_cpu_load_average_10s', 'node_network_transmit_fifo_total', 'node_network_iface_link', 'node_memory_DirectMap2M_bytes', 
              'node_memory_HardwareCorrupted_bytes', 'node_disk_io_now', 'node_memory_WritebackTmp_bytes']

# 34
metric_tags_keys=['__name__','beta_kubernetes_io_arch','beta_kubernetes_io_os','container','id',
                  'image','instance','job','kubernetes_io_arch','kubernetes_io_hostname',
                  'kubernetes_io_os','name','namespace','pod','metric_name',
                  'node_role_kubernetes_io_master','cpu','interface','app','controller_revision_hash',
                  'kubernetes_namespace','kubernetes_pod_name','mode','pod_template_generation','kubernetes_name',
                  'device','major','minor','address','broadcast',
                  'duplex','ifalias','operstate','service_name']

# 94
metric_tag_node_cpu2=['node_cpu_seconds_total', 'node_cpu_guest_seconds_total']
metric_tag_node_disk12=['node_disk_reads_merged_total', 'node_disk_io_time_weighted_seconds_total', 'node_disk_info', 'node_disk_reads_completed_total', 'node_disk_write_time_seconds_total', 
                       'node_disk_read_bytes_total', 'node_disk_writes_completed_total', 'node_disk_written_bytes_total', 'node_disk_io_time_seconds_total', 'node_disk_read_time_seconds_total', 
                       'node_disk_writes_merged_total', 'node_disk_io_now']
metric_tag_node_network32=['node_network_transmit_queue_length', 'node_network_transmit_errs_total', 'node_network_mtu_bytes', 'node_network_protocol_type', 'node_network_address_assign_type', 
                           'node_network_carrier_changes_total', 'node_network_net_dev_group', 'node_network_device_id', 'node_network_flags', 'node_network_iface_id', 
                           'node_network_iface_link_mode', 'node_network_transmit_carrier_total', 'node_network_transmit_compressed_total', 'node_network_speed_bytes', 'node_network_receive_multicast_total', 
                           'node_network_info', 'node_network_receive_packets_total', 'node_network_transmit_colls_total', 'node_network_receive_frame_total', 'node_network_transmit_packets_total', 
                           'node_network_receive_drop_total', 'node_network_transmit_drop_total', 'node_network_receive_fifo_total', 'node_network_dormant', 'node_network_transmit_bytes_total', 
                           'node_network_receive_compressed_total', 'node_network_receive_errs_total', 'node_network_up', 'node_network_carrier', 'node_network_receive_bytes_total', 
                           'node_network_transmit_fifo_total', 'node_network_iface_link']
metric_tag_node_memory45=['node_memory_Active_file_bytes', 'node_memory_Slab_bytes', 'node_memory_Inactive_anon_bytes', 'node_memory_DirectMap4k_bytes', 'node_memory_SwapFree_bytes', 
                          'node_memory_Inactive_file_bytes', 'node_memory_Inactive_bytes', 'node_memory_HugePages_Rsvd', 'node_memory_Mlocked_bytes', 'node_memory_Active_bytes', 
                          'node_memory_Unevictable_bytes', 'node_memory_Shmem_bytes', 'node_memory_MemAvailable_bytes', 'node_memory_VmallocChunk_bytes', 'node_memory_Mapped_bytes', 
                          'node_memory_SReclaimable_bytes', 'node_memory_Buffers_bytes', 'node_memory_Committed_AS_bytes', 'node_memory_Dirty_bytes', 'node_memory_SwapCached_bytes', 
                          'node_memory_Cached_bytes', 'node_memory_PageTables_bytes', 'node_memory_SUnreclaim_bytes', 'node_memory_MemTotal_bytes', 'node_memory_CommitLimit_bytes', 
                          'node_memory_KernelStack_bytes', 'node_memory_Hugepagesize_bytes', 'node_memory_HugePages_Total', 'node_memory_Writeback_bytes', 'node_memory_VmallocUsed_bytes', 
                          'node_memory_HugePages_Free', 'node_memory_MemFree_bytes', 'node_memory_CmaTotal_bytes', 'node_memory_NFS_Unstable_bytes', 'node_memory_AnonHugePages_bytes', 
                          'node_memory_Bounce_bytes', 'node_memory_AnonPages_bytes', 'node_memory_VmallocTotal_bytes', 'node_memory_SwapTotal_bytes', 'node_memory_HugePages_Surp', 
                          'node_memory_CmaFree_bytes', 'node_memory_Active_anon_bytes', 'node_memory_DirectMap2M_bytes', 'node_memory_HardwareCorrupted_bytes', 'node_memory_WritebackTmp_bytes']
metric_tag_node_load3=['node_load1', 'node_load15', 'node_load5']
metric_tag_node_94=metric_tag_node_cpu2+metric_tag_node_disk12+metric_tag_node_network32+metric_tag_node_memory45+metric_tag_node_load3

# 44
metric_tag_node_instance44=['10.60.216.19:9100', '10.60.41.120:9100', '10.60.82.102:9100', '10.60.84.180:9100', 
                            '10.60.198.27:9100', '10.60.70.72:9100', '10.60.61.196:9100', '10.60.91.167:9100', 
                            '10.60.252.14:9100', '10.60.115.117:9100', '10.60.108.232:9100', '10.60.169.14:9100', 
                            '10.60.74.40:9100', '10.60.154.38:9100', '10.60.188.219:9100', '10.60.75.28:9100', 
                            '10.60.112.68:9100', '10.60.16.44:9100', '10.60.82.175:9100', '10.60.151.113:9100', 
                            '10.60.226.20:9100', '10.60.116.144:9100', '10.60.238.155:9100', '10.60.151.97:9100', 
                            '10.60.223.222:9100', '10.60.148.246:9100', '10.60.147.171:9100', '10.60.225.136:9100', 
                            '10.60.201.193:9100', '10.60.22.248:9100', '10.60.181.41:9100', '10.60.204.152:9100', 
                            '10.60.84.226:9100', '10.60.7.88:9100', '10.60.90.184:9100', '10.60.27.185:9100', 
                            '10.60.72.159:9100', '10.60.137.240:9100', '10.60.228.124:9100', '10.60.50.101:9100', 
                            '10.60.130.74:9100', '10.60.38.72:9100', '10.60.105.33:9100', '10.60.101.252:9100']

metric_tags_4= ['cpm','resp_time','error_count','success_rate']
#40
metric_tag_service_name_test=['9adcafa7a8054bea979a5a3ed4694ad7', '4bb63a99a49f42ae9c3d9139d4d4482c', 'd7189bdd1e8b43b59a5f6284bb89798a', '4a5efe1fa59347b7883461b2debb3e27', '6cc7dc7bb5fa4327a20c883ab00ab2fe', 
                              '32cb0cd3d5924701ad10586ee356d8fe', '53f1acb37db941b8b9c77dfefecb157b', '122ec12af3744773b9b04c6c8e929711', '6ddef64b548f45b49da90a040777bb4f', '0bea01d71c834b0bba9aaa41e9884cf8', 
                              '4287f5cca47742008a8fb965908e5dea', '00c1ba198361424c9597328ea33d0d15', '8b92c25a69954a66bb13dcaf394b4499', 'a8d733e6a7894d46bb9c7f1b58bbc192', '15b7f6577fec4c16b01ee2e053b1f201', 
                              '2170e75abdf54178afcd5ffecb387eee', 'c453a975de5148e4b1c47be258a646c9', '0d1304f1f40743dea03be55bca96c32b', '883cd056805347a5b0482e2cbf1cd97f', '192756d8271842a9a06b4252ff4e5e7b', 
                              '848ec944cf1a4391be09dd24c105aea9', '68fbe2fc0a48404c987dcb108906349a', 'f1023ca9976e4a5eaaaaed244acd2f4a', 'faf90b12d1cf478e810172eb6aced658', '3d82f4ad7f114cbdbd469fc897b001a1', 
                              '14eb4630112b4ce9bd88d93104b4570e', 'ea4bdf00441c4157a99a9c72bb7f4eb2', '72d37257c88046aba2283fd7e602dfae', '9ab2c6acd4ca4925955bd41e23016f5c', '63555cbf9b6341a99dd1dc5494158a28', 
                              '6566a72fb2a1461ca692c58dc88dff90', '03d1f58da52d49dbb815cda9be061d25', 'fabe1018f1c2459a84864ddfecb30f3a', '8b3eee3cc4fe4568b5ba4125c1a4047f', 'ad8fb26af5154e5b9dc4fb1f609282fb', 
                              'e572b4f9245643138661a71d17838452', 'b89ca58ac95c4675af100efcb7dd485c', '36c4ac32f7504f13b7aef941de9ecc81', '8b18231981e0440488bbac370b1464cf', '05be78908e3c4818b1caa00b71d8bb11']



# 15
metric_tags_15=['container_network_transmit_errors_total', 'container_network_receive_packets_dropped_total', 'container_cpu_system_seconds_total', 'container_network_transmit_bytes_total', 'container_network_transmit_packets_dropped_total', 
                'container_network_receive_bytes_total', 'container_cpu_cfs_throttled_seconds_total', 'container_network_transmit_packets_total', 'container_network_receive_errors_total', 'container_network_receive_packets_total', 
                'container_cpu_cfs_throttled_periods_total', 'container_cpu_user_seconds_total', 'container_cpu_cfs_periods_total', 'container_cpu_usage_seconds_total', 'container_cpu_load_average_10s']
metric_tag_contain_cpu7=['container_cpu_system_seconds_total', 'container_cpu_cfs_throttled_seconds_total', 'container_cpu_cfs_throttled_periods_total', 'container_cpu_user_seconds_total', 'container_cpu_cfs_periods_total', 
                        'container_cpu_usage_seconds_total', 'container_cpu_load_average_10s']
metric_tag_contain_network8=['container_network_transmit_errors_total', 'container_network_receive_packets_dropped_total', 'container_network_transmit_bytes_total', 'container_network_transmit_packets_dropped_total', 'container_network_receive_bytes_total', 
                            'container_network_transmit_packets_total', 'container_network_receive_errors_total', 'container_network_receive_packets_total']

metric_tag_instance_nsdefault=['node-worker1', 'node-worker3', 'node-worker2', 'node-master']

metric_container_80=['', '8b18231981e0440488bbac370b1464cf', 'kube-state-metrics', 'prometheus-alertmanager', 'POD', 
                     '6566a72fb2a1461ca692c58dc88dff90', '0bea01d71c834b0bba9aaa41e9884cf8', 'chaosblade-operator', '00c1ba198361424c9597328ea33d0d15', 'chaosblade-tool', 
                     '53f1acb37db941b8b9c77dfefecb157b', 'e572b4f9245643138661a71d17838452', 'admission-webhook', '122ec12af3744773b9b04c6c8e929711', 'b89ca58ac95c4675af100efcb7dd485c', 
                     'kube-proxy', 'openebs-snapshot-controller', 'f1023ca9976e4a5eaaaaed244acd2f4a', 'grafana-core', 'skywalking', 
                     '4bb63a99a49f42ae9c3d9139d4d4482c', '03d1f58da52d49dbb815cda9be061d25', '3d82f4ad7f114cbdbd469fc897b001a1', '15b7f6577fec4c16b01ee2e053b1f201', '0d1304f1f40743dea03be55bca96c32b', 
                     'openebs-snapshot-provisioner', 'openebs-provisioner', '72d37257c88046aba2283fd7e602dfae', 'kube-apiserver', 'node-exporter', 
                     '32cb0cd3d5924701ad10586ee356d8fe', 'fabe1018f1c2459a84864ddfecb30f3a', 'openebs-ndm-operator', 'mysql', 'prometheus', 
                     '26490d5901ce44aabcbc33dbc5925adc', 'openebs-ndm', 'c453a975de5148e4b1c47be258a646c9', '883cd056805347a5b0482e2cbf1cd97f', '63555cbf9b6341a99dd1dc5494158a28', 
                     '05be78908e3c4818b1caa00b71d8bb11', 'k8snacos', 'etcd', 'ad8fb26af5154e5b9dc4fb1f609282fb', 'alertsnitch-mysql', 
                     '14eb4630112b4ce9bd88d93104b4570e', '69ebe09877684907811e840ba66ea578', 'kube-controller-manager', '2170e75abdf54178afcd5ffecb387eee', 'coredns', 
                     'alertsnitch', 'e97a387ed0204878b0660f0090bfacd6', 'openebs-localpv-provisioner', '6ddef64b548f45b49da90a040777bb4f', '177495fd11344929857a237816afbb41', 
                     'a8d733e6a7894d46bb9c7f1b58bbc192', 'skywalking-ui', '6adddd8cff25463a97ee86a9739c86d3', '9adcafa7a8054bea979a5a3ed4694ad7', '33f2b505c8a5435988a20787fbe522fc', 
                     '9ab2c6acd4ca4925955bd41e23016f5c', 'kube-scheduler', '848ec944cf1a4391be09dd24c105aea9', '192756d8271842a9a06b4252ff4e5e7b', 'ea4bdf00441c4157a99a9c72bb7f4eb2', 
                     '68fbe2fc0a48404c987dcb108906349a', '8b92c25a69954a66bb13dcaf394b4499', 'rabbitmq', '4a5efe1fa59347b7883461b2debb3e27', '4287f5cca47742008a8fb965908e5dea', 
                     '6cc7dc7bb5fa4327a20c883ab00ab2fe', 'd7189bdd1e8b43b59a5f6284bb89798a', '8b3eee3cc4fe4568b5ba4125c1a4047f', 'openebs-apiserver', '36c4ac32f7504f13b7aef941de9ecc81', 
                     'kube-flannel', 'faf90b12d1cf478e810172eb6aced658', 'xenon', 'elasticsearch', 'slowlog']

metric_pod80=['', 'alertsnitch-mysql', '9ab2c6acd4ca4925955bd41e23016f5c-79c99665c8', 'openebs-provisioner', '2170e75abdf54178afcd5ffecb387eee-ddbfb8846', 
              '8b18231981e0440488bbac370b1464cf-68c85f66f7', 'openebs-apiserver', 'e572b4f9245643138661a71d17838452-76d6fc4c', '3d82f4ad7f114cbdbd469fc897b001a1-646f76f48c', '15b7f6577fec4c16b01ee2e053b1f201-74d798f84b', 
              '53f1acb37db941b8b9c77dfefecb157b-67c4bbcb66', '33f2b505c8a5435988a20787fbe522fc-6695d44fb', 'b89ca58ac95c4675af100efcb7dd485c-7865446cb9', 'ea4bdf00441c4157a99a9c72bb7f4eb2-5bc6cf98b5', 'grafana-core', 
              'ad8fb26af5154e5b9dc4fb1f609282fb-5f9c4dc84c', '6adddd8cff25463a97ee86a9739c86d3-5d9898c9bb', 'kube-state', 'rabbitmq-5d8589794f', 'skywalking-5c9f58c947', 'f1023ca9976e4a5eaaaaed244acd2f4a-7f86c4c4bb', 
              'e97a387ed0204878b0660f0090bfacd6-57f599655b', 'kube-flannel', '6566a72fb2a1461ca692c58dc88dff90-6944cd495c', 'd7189bdd1e8b43b59a5f6284bb89798a-fd57867bf', 'fabe1018f1c2459a84864ddfecb30f3a-7676d54bb', 
              '8b92c25a69954a66bb13dcaf394b4499-698db669f8', '63555cbf9b6341a99dd1dc5494158a28-697dddf596', '4bb63a99a49f42ae9c3d9139d4d4482c-774f756cc9', 'elasticsearch-58cd769777', '9adcafa7a8054bea979a5a3ed4694ad7-6747cccb7f', 
              '177495fd11344929857a237816afbb41-747cfdd897', 'kube-controller', 'c453a975de5148e4b1c47be258a646c9-7999d9d4f8', 'coredns-6955765f44', 'prometheus-989b58f9', 'nacos-1', 'skywalking-ui', 'kube-scheduler', 'openebs-ndm', 
              'alertmanager-777cf86864', '05be78908e3c4818b1caa00b71d8bb11-5bcfffdd99', 'ab1c1a0046754e49b4fdf64708124365', 'chaosblade-operator', '122ec12af3744773b9b04c6c8e929711-84f64bf856', '883cd056805347a5b0482e2cbf1cd97f-5f5996475f', 
              'faf90b12d1cf478e810172eb6aced658-64b8b7dd6f', '6cc7dc7bb5fa4327a20c883ab00ab2fe-675b8fd8b', 'openebs-admission', '72d37257c88046aba2283fd7e602dfae-7cfb745bd5', 'nacosdb-mysql', 'kube-apiserver', 'nacos-0', 
              '68fbe2fc0a48404c987dcb108906349a-67f9d6978c', '03d1f58da52d49dbb815cda9be061d25-c8949d9d9', 'openebs-localpv', '4287f5cca47742008a8fb965908e5dea-744b568489', 'kube-proxy', 'node-exporter', 'a8d733e6a7894d46bb9c7f1b58bbc192-857c85f6b5', 
              'c019762f3cbd493cb4dc4443eec2c273', 'alertsnitch-7bf59b5fbf', '0d1304f1f40743dea03be55bca96c32b-796d45c769', '26490d5901ce44aabcbc33dbc5925adc-66ddf858cc', '4a5efe1fa59347b7883461b2debb3e27-7c59bfb675', 'nacos-2', '7f1cbe89dd024b6ebbf5556426b34acf', 
              '00c1ba198361424c9597328ea33d0d15-7f5656d7d9', '848ec944cf1a4391be09dd24c105aea9-6dd6cc5cf', '6ddef64b548f45b49da90a040777bb4f-5484c9d98f', '192756d8271842a9a06b4252ff4e5e7b-b9d8b4895', '69ebe09877684907811e840ba66ea578-5969f446c5', 
              'openebs-snapshot', 'etcd-node', '36c4ac32f7504f13b7aef941de9ecc81-cb495dcfc', 'chaosblade-tool', '32cb0cd3d5924701ad10586ee356d8fe-bff9d9468', '14eb4630112b4ce9bd88d93104b4570e-c4575c8f5', '0bea01d71c834b0bba9aaa41e9884cf8-58fdd89ffc',
              '8b3eee3cc4fe4568b5ba4125c1a4047f-b57494557']

metric_interface5=['eth0','eth1','eth2','cni0','flannel.1']

metric_contain_net8_pod2interface={'': ['eth0', 'eth1', 'eth2', 'flannel.1', 'cni0'], 'openebs-admission': ['eth0'], '6adddd8cff25463a97ee86a9739c86d3-5d9898c9bb': ['eth0'], 
                                  'kube-flannel': ['cni0', 'eth0', 'eth1', 'eth2', 'flannel.1'], '0bea01d71c834b0bba9aaa41e9884cf8-58fdd89ffc': ['eth0'], 
                                  'c453a975de5148e4b1c47be258a646c9-7999d9d4f8': ['eth0'], 'kube-state': ['eth0'], '4bb63a99a49f42ae9c3d9139d4d4482c-774f756cc9': ['eth0'], 
                                  '8b92c25a69954a66bb13dcaf394b4499-698db669f8': ['eth0'], 'faf90b12d1cf478e810172eb6aced658-64b8b7dd6f': ['eth0'], 'nacosdb-mysql': ['eth0'], 
                                  '68fbe2fc0a48404c987dcb108906349a-67f9d6978c': ['eth0'], 'coredns-6955765f44': ['eth0'], 'alertsnitch-mysql': ['eth0'], 
                                  '26490d5901ce44aabcbc33dbc5925adc-66ddf858cc': ['eth0'], '8b3eee3cc4fe4568b5ba4125c1a4047f-b57494557': ['eth0'], 
                                  '33f2b505c8a5435988a20787fbe522fc-6695d44fb': ['eth0'], 'skywalking-ui': ['eth0'], '72d37257c88046aba2283fd7e602dfae-7cfb745bd5': ['eth0'], 
                                  'openebs-snapshot': ['eth0'], 'node-exporter': ['cni0', 'eth0', 'eth1', 'eth2', 'flannel.1'], 'chaosblade-tool': ['eth0', 'eth1', 'eth2', 'flannel.1', 'cni0'], 
                                  '848ec944cf1a4391be09dd24c105aea9-6dd6cc5cf': ['eth0'], '32cb0cd3d5924701ad10586ee356d8fe-bff9d9468': ['eth0'], 'nacos-0': ['eth0'], 'nacos-2': ['eth0'], 
                                  'kube-controller': ['eth0', 'eth1', 'eth2', 'flannel.1'], 'b89ca58ac95c4675af100efcb7dd485c-7865446cb9': ['eth0'], '7f1cbe89dd024b6ebbf5556426b34acf': ['eth0'], 
                                  '0d1304f1f40743dea03be55bca96c32b-796d45c769': ['eth0'], '53f1acb37db941b8b9c77dfefecb157b-67c4bbcb66': ['eth0'], 'chaosblade-operator': ['eth0'], 
                                  '8b18231981e0440488bbac370b1464cf-68c85f66f7': ['eth0'], '177495fd11344929857a237816afbb41-747cfdd897': ['eth0'], '69ebe09877684907811e840ba66ea578-5969f446c5': ['eth0'], 
                                  '192756d8271842a9a06b4252ff4e5e7b-b9d8b4895': ['eth0'], 'openebs-apiserver': ['eth0'], 'openebs-provisioner': ['eth0'], '4287f5cca47742008a8fb965908e5dea-744b568489': ['eth0'], 
                                  '6ddef64b548f45b49da90a040777bb4f-5484c9d98f': ['eth0'], '2170e75abdf54178afcd5ffecb387eee-ddbfb8846': ['eth0'], 'f1023ca9976e4a5eaaaaed244acd2f4a-7f86c4c4bb': ['eth0'], 
                                  'kube-apiserver': ['eth0', 'eth1', 'eth2', 'flannel.1'], '36c4ac32f7504f13b7aef941de9ecc81-cb495dcfc': ['eth0'], 'etcd-node': ['eth0', 'eth1', 'eth2', 'flannel.1'], 
                                  'c019762f3cbd493cb4dc4443eec2c273': ['eth0'], 'grafana-core': ['eth0'], '3d82f4ad7f114cbdbd469fc897b001a1-646f76f48c': ['eth0'], 
                                  '63555cbf9b6341a99dd1dc5494158a28-697dddf596': ['eth0'], '03d1f58da52d49dbb815cda9be061d25-c8949d9d9': ['eth0'], 'alertmanager-777cf86864': ['eth0'], 
                                  '122ec12af3744773b9b04c6c8e929711-84f64bf856': ['eth0'], 'a8d733e6a7894d46bb9c7f1b58bbc192-857c85f6b5': ['eth0'], 'kube-scheduler': ['eth0', 'eth1', 'eth2', 'flannel.1'], 
                                  'openebs-ndm': ['cni0', 'eth0', 'eth1', 'eth2', 'flannel.1'], 'kube-proxy': ['cni0', 'eth0', 'eth1', 'eth2', 'flannel.1'], 'ab1c1a0046754e49b4fdf64708124365': ['eth0'], 
                                  'rabbitmq-5d8589794f': ['eth0'], 'prometheus-989b58f9': ['eth0'], 'alertsnitch-7bf59b5fbf': ['eth0'], '6566a72fb2a1461ca692c58dc88dff90-6944cd495c': ['eth0'], 
                                  '00c1ba198361424c9597328ea33d0d15-7f5656d7d9': ['eth0'], 'nacos-1': ['eth0'], '4a5efe1fa59347b7883461b2debb3e27-7c59bfb675': ['eth0'], 'skywalking-5c9f58c947': ['eth0'], 
                                  '6cc7dc7bb5fa4327a20c883ab00ab2fe-675b8fd8b': ['eth0'], 'd7189bdd1e8b43b59a5f6284bb89798a-fd57867bf': ['eth0'], 'ad8fb26af5154e5b9dc4fb1f609282fb-5f9c4dc84c': ['eth0'], 
                                  '883cd056805347a5b0482e2cbf1cd97f-5f5996475f': ['eth0'], 'openebs-localpv': ['eth0'], '9adcafa7a8054bea979a5a3ed4694ad7-6747cccb7f': ['eth0'], 'elasticsearch-58cd769777': ['eth0'], 
                                  'e97a387ed0204878b0660f0090bfacd6-57f599655b': ['eth0'], 'fabe1018f1c2459a84864ddfecb30f3a-7676d54bb': ['eth0'], '05be78908e3c4818b1caa00b71d8bb11-5bcfffdd99': ['eth0'], 
                                  'e572b4f9245643138661a71d17838452-76d6fc4c': ['eth0'], '14eb4630112b4ce9bd88d93104b4570e-c4575c8f5': ['eth0'], 'ea4bdf00441c4157a99a9c72bb7f4eb2-5bc6cf98b5': ['eth0'], 
                                  '15b7f6577fec4c16b01ee2e053b1f201-74d798f84b': ['eth0'], '9ab2c6acd4ca4925955bd41e23016f5c-79c99665c8': ['eth0']}

metric_tag_cpu7_2container_3_pod57=['7f1cbe89dd024b6ebbf5556426b34acf', '14eb4630112b4ce9bd88d93104b4570e-c4575c8f5', '4a5efe1fa59347b7883461b2debb3e27-7c59bfb675', '33f2b505c8a5435988a20787fbe522fc-6695d44fb', 'grafana-core', 
                                    '0d1304f1f40743dea03be55bca96c32b-796d45c769', '32cb0cd3d5924701ad10586ee356d8fe-bff9d9468', '6566a72fb2a1461ca692c58dc88dff90-6944cd495c', '122ec12af3744773b9b04c6c8e929711-84f64bf856', 'skywalking-5c9f58c947', 
                                    '4bb63a99a49f42ae9c3d9139d4d4482c-774f756cc9', 'e572b4f9245643138661a71d17838452-76d6fc4c', '00c1ba198361424c9597328ea33d0d15-7f5656d7d9', '9adcafa7a8054bea979a5a3ed4694ad7-6747cccb7f', '15b7f6577fec4c16b01ee2e053b1f201-74d798f84b', 
                                    '3d82f4ad7f114cbdbd469fc897b001a1-646f76f48c', 'b89ca58ac95c4675af100efcb7dd485c-7865446cb9', 'a8d733e6a7894d46bb9c7f1b58bbc192-857c85f6b5', '53f1acb37db941b8b9c77dfefecb157b-67c4bbcb66', '8b92c25a69954a66bb13dcaf394b4499-698db669f8', 
                                    'd7189bdd1e8b43b59a5f6284bb89798a-fd57867bf', '8b18231981e0440488bbac370b1464cf-68c85f66f7', '69ebe09877684907811e840ba66ea578-5969f446c5', '883cd056805347a5b0482e2cbf1cd97f-5f5996475f', '6ddef64b548f45b49da90a040777bb4f-5484c9d98f', 
                                    '0bea01d71c834b0bba9aaa41e9884cf8-58fdd89ffc', 'nacosdb-mysql', '03d1f58da52d49dbb815cda9be061d25-c8949d9d9', '68fbe2fc0a48404c987dcb108906349a-67f9d6978c', '72d37257c88046aba2283fd7e602dfae-7cfb745bd5', 
                                    'fabe1018f1c2459a84864ddfecb30f3a-7676d54bb', '8b3eee3cc4fe4568b5ba4125c1a4047f-b57494557', '6adddd8cff25463a97ee86a9739c86d3-5d9898c9bb', 'skywalking-ui', '26490d5901ce44aabcbc33dbc5925adc-66ddf858cc', 
                                    'ad8fb26af5154e5b9dc4fb1f609282fb-5f9c4dc84c', '63555cbf9b6341a99dd1dc5494158a28-697dddf596', 'prometheus-989b58f9', 'ab1c1a0046754e49b4fdf64708124365', 'c453a975de5148e4b1c47be258a646c9-7999d9d4f8', 
                                    '6cc7dc7bb5fa4327a20c883ab00ab2fe-675b8fd8b', '2170e75abdf54178afcd5ffecb387eee-ddbfb8846', '192756d8271842a9a06b4252ff4e5e7b-b9d8b4895', '848ec944cf1a4391be09dd24c105aea9-6dd6cc5cf', 'elasticsearch-58cd769777', 
                                    '9ab2c6acd4ca4925955bd41e23016f5c-79c99665c8', '36c4ac32f7504f13b7aef941de9ecc81-cb495dcfc', 'e97a387ed0204878b0660f0090bfacd6-57f599655b', 'kube-flannel', '177495fd11344929857a237816afbb41-747cfdd897', 
                                    'faf90b12d1cf478e810172eb6aced658-64b8b7dd6f', 'alertmanager-777cf86864', 'f1023ca9976e4a5eaaaaed244acd2f4a-7f86c4c4bb', 'ea4bdf00441c4157a99a9c72bb7f4eb2-5bc6cf98b5', '05be78908e3c4818b1caa00b71d8bb11-5bcfffdd99', 
                                    '4287f5cca47742008a8fb965908e5dea-744b568489', 'c019762f3cbd493cb4dc4443eec2c273']
metric_tag_cpu7_3container_4_pod80=['', 'alertsnitch-mysql', '9ab2c6acd4ca4925955bd41e23016f5c-79c99665c8', 'openebs-provisioner', '2170e75abdf54178afcd5ffecb387eee-ddbfb8846', 
                                    '8b18231981e0440488bbac370b1464cf-68c85f66f7', 'openebs-apiserver', 'e572b4f9245643138661a71d17838452-76d6fc4c', '3d82f4ad7f114cbdbd469fc897b001a1-646f76f48c', '15b7f6577fec4c16b01ee2e053b1f201-74d798f84b', 
                                    '53f1acb37db941b8b9c77dfefecb157b-67c4bbcb66', '33f2b505c8a5435988a20787fbe522fc-6695d44fb', 'b89ca58ac95c4675af100efcb7dd485c-7865446cb9', 'ea4bdf00441c4157a99a9c72bb7f4eb2-5bc6cf98b5', 'grafana-core', 
                                    'ad8fb26af5154e5b9dc4fb1f609282fb-5f9c4dc84c', '6adddd8cff25463a97ee86a9739c86d3-5d9898c9bb', 'kube-state', 'rabbitmq-5d8589794f', 'skywalking-5c9f58c947', 'f1023ca9976e4a5eaaaaed244acd2f4a-7f86c4c4bb', 
                                    'e97a387ed0204878b0660f0090bfacd6-57f599655b', 'kube-flannel', '6566a72fb2a1461ca692c58dc88dff90-6944cd495c', 'd7189bdd1e8b43b59a5f6284bb89798a-fd57867bf', 'fabe1018f1c2459a84864ddfecb30f3a-7676d54bb', 
                                    '8b92c25a69954a66bb13dcaf394b4499-698db669f8', '63555cbf9b6341a99dd1dc5494158a28-697dddf596', '4bb63a99a49f42ae9c3d9139d4d4482c-774f756cc9', 'elasticsearch-58cd769777', '9adcafa7a8054bea979a5a3ed4694ad7-6747cccb7f', 
                                    '177495fd11344929857a237816afbb41-747cfdd897', 'kube-controller', 'c453a975de5148e4b1c47be258a646c9-7999d9d4f8', 'coredns-6955765f44', 'prometheus-989b58f9', 'nacos-1', 'skywalking-ui', 'kube-scheduler', 'openebs-ndm', 
                                    'alertmanager-777cf86864', '05be78908e3c4818b1caa00b71d8bb11-5bcfffdd99', 'ab1c1a0046754e49b4fdf64708124365', 'chaosblade-operator', '122ec12af3744773b9b04c6c8e929711-84f64bf856', '883cd056805347a5b0482e2cbf1cd97f-5f5996475f', 
                                    'faf90b12d1cf478e810172eb6aced658-64b8b7dd6f', '6cc7dc7bb5fa4327a20c883ab00ab2fe-675b8fd8b', 'openebs-admission', '72d37257c88046aba2283fd7e602dfae-7cfb745bd5', 'nacosdb-mysql', 'kube-apiserver', 'nacos-0', 
                                    '68fbe2fc0a48404c987dcb108906349a-67f9d6978c', '03d1f58da52d49dbb815cda9be061d25-c8949d9d9', 'openebs-localpv', '4287f5cca47742008a8fb965908e5dea-744b568489', 'kube-proxy', 'node-exporter', 'a8d733e6a7894d46bb9c7f1b58bbc192-857c85f6b5', 
                                    'c019762f3cbd493cb4dc4443eec2c273', 'alertsnitch-7bf59b5fbf', '0d1304f1f40743dea03be55bca96c32b-796d45c769', '26490d5901ce44aabcbc33dbc5925adc-66ddf858cc', '4a5efe1fa59347b7883461b2debb3e27-7c59bfb675', 'nacos-2', '7f1cbe89dd024b6ebbf5556426b34acf', 
                                    '00c1ba198361424c9597328ea33d0d15-7f5656d7d9', '848ec944cf1a4391be09dd24c105aea9-6dd6cc5cf', '6ddef64b548f45b49da90a040777bb4f-5484c9d98f', '192756d8271842a9a06b4252ff4e5e7b-b9d8b4895', '69ebe09877684907811e840ba66ea578-5969f446c5', 
                                    'openebs-snapshot', 'etcd-node', '36c4ac32f7504f13b7aef941de9ecc81-cb495dcfc', 'chaosblade-tool', '32cb0cd3d5924701ad10586ee356d8fe-bff9d9468', '14eb4630112b4ce9bd88d93104b4570e-c4575c8f5', '0bea01d71c834b0bba9aaa41e9884cf8-58fdd89ffc',
                                    '8b3eee3cc4fe4568b5ba4125c1a4047f-b57494557']

metric_contain_cpu7_instance73=['alertsnitch-mysql', '9ab2c6acd4ca4925955bd41e23016f5c-79c99665c8', 'openebs-provisioner', '2170e75abdf54178afcd5ffecb387eee-ddbfb8846', 
                                '8b18231981e0440488bbac370b1464cf-68c85f66f7', 'openebs-apiserver', 'e572b4f9245643138661a71d17838452-76d6fc4c', 
                                '3d82f4ad7f114cbdbd469fc897b001a1-646f76f48c', '15b7f6577fec4c16b01ee2e053b1f201-74d798f84b', '53f1acb37db941b8b9c77dfefecb157b-67c4bbcb66', 
                                '33f2b505c8a5435988a20787fbe522fc-6695d44fb', 'b89ca58ac95c4675af100efcb7dd485c-7865446cb9', 'ea4bdf00441c4157a99a9c72bb7f4eb2-5bc6cf98b5', 
                                'grafana-core', 'ad8fb26af5154e5b9dc4fb1f609282fb-5f9c4dc84c', '6adddd8cff25463a97ee86a9739c86d3-5d9898c9bb', 'kube-state', 'rabbitmq-5d8589794f', 
                                'skywalking-5c9f58c947', 'f1023ca9976e4a5eaaaaed244acd2f4a-7f86c4c4bb', 'e97a387ed0204878b0660f0090bfacd6-57f599655b', 
                                '6566a72fb2a1461ca692c58dc88dff90-6944cd495c', 'd7189bdd1e8b43b59a5f6284bb89798a-fd57867bf', 'fabe1018f1c2459a84864ddfecb30f3a-7676d54bb', 
                                '8b92c25a69954a66bb13dcaf394b4499-698db669f8', '63555cbf9b6341a99dd1dc5494158a28-697dddf596', '4bb63a99a49f42ae9c3d9139d4d4482c-774f756cc9', 
                                'elasticsearch-58cd769777', '9adcafa7a8054bea979a5a3ed4694ad7-6747cccb7f', '177495fd11344929857a237816afbb41-747cfdd897', 'kube-controller', 
                                'c453a975de5148e4b1c47be258a646c9-7999d9d4f8', 'coredns-6955765f44', 'prometheus-989b58f9', 'nacos-1', 'skywalking-ui', 'kube-scheduler', 
                                'alertmanager-777cf86864', '05be78908e3c4818b1caa00b71d8bb11-5bcfffdd99', 'ab1c1a0046754e49b4fdf64708124365', 'chaosblade-operator', 
                                '122ec12af3744773b9b04c6c8e929711-84f64bf856', '883cd056805347a5b0482e2cbf1cd97f-5f5996475f', 'faf90b12d1cf478e810172eb6aced658-64b8b7dd6f', 
                                '6cc7dc7bb5fa4327a20c883ab00ab2fe-675b8fd8b', 'openebs-admission', '72d37257c88046aba2283fd7e602dfae-7cfb745bd5', 'kube-apiserver', 'nacos-0', 
                                '68fbe2fc0a48404c987dcb108906349a-67f9d6978c', '03d1f58da52d49dbb815cda9be061d25-c8949d9d9', 'openebs-localpv', 
                                '4287f5cca47742008a8fb965908e5dea-744b568489', 'a8d733e6a7894d46bb9c7f1b58bbc192-857c85f6b5', 'c019762f3cbd493cb4dc4443eec2c273', 'alertsnitch-7bf59b5fbf',
                                '0d1304f1f40743dea03be55bca96c32b-796d45c769', '26490d5901ce44aabcbc33dbc5925adc-66ddf858cc', '4a5efe1fa59347b7883461b2debb3e27-7c59bfb675', 'nacos-2', 
                                '7f1cbe89dd024b6ebbf5556426b34acf', '00c1ba198361424c9597328ea33d0d15-7f5656d7d9', '848ec944cf1a4391be09dd24c105aea9-6dd6cc5cf', 
                                '6ddef64b548f45b49da90a040777bb4f-5484c9d98f', '192756d8271842a9a06b4252ff4e5e7b-b9d8b4895', '69ebe09877684907811e840ba66ea578-5969f446c5', 
                                'openebs-snapshot', 'etcd-node', '36c4ac32f7504f13b7aef941de9ecc81-cb495dcfc', '32cb0cd3d5924701ad10586ee356d8fe-bff9d9468', 
                                '14eb4630112b4ce9bd88d93104b4570e-c4575c8f5', '0bea01d71c834b0bba9aaa41e9884cf8-58fdd89ffc', '8b3eee3cc4fe4568b5ba4125c1a4047f-b57494557']
metric_contain_cpu7_4instance7=['kube-flannel', '', 'nacosdb-mysql','openebs-ndm', 'kube-proxy', 'node-exporter', 'chaosblade-tool']

metric_tag_cpu7_2container_3=['container_cpu_cfs_periods_total','container_cpu_cfs_throttled_periods_total','container_cpu_cfs_throttled_seconds_total']
metric_tag_cpu7_2container_3_4instance2=['nacosdb-mysql', 'kube-flannel']
metric_tag_cpu7_2container_3_emptyct_pod53=['14eb4630112b4ce9bd88d93104b4570e-c4575c8f5', '4a5efe1fa59347b7883461b2debb3e27-7c59bfb675', '33f2b505c8a5435988a20787fbe522fc-6695d44fb', 'grafana-core', 
                                        '0d1304f1f40743dea03be55bca96c32b-796d45c769', '32cb0cd3d5924701ad10586ee356d8fe-bff9d9468', '6566a72fb2a1461ca692c58dc88dff90-6944cd495c', 
                                        '122ec12af3744773b9b04c6c8e929711-84f64bf856', 'skywalking-5c9f58c947', '4bb63a99a49f42ae9c3d9139d4d4482c-774f756cc9', 
                                        'e572b4f9245643138661a71d17838452-76d6fc4c', '00c1ba198361424c9597328ea33d0d15-7f5656d7d9', '9adcafa7a8054bea979a5a3ed4694ad7-6747cccb7f', 
                                        '15b7f6577fec4c16b01ee2e053b1f201-74d798f84b', '3d82f4ad7f114cbdbd469fc897b001a1-646f76f48c', 'b89ca58ac95c4675af100efcb7dd485c-7865446cb9', 
                                        'a8d733e6a7894d46bb9c7f1b58bbc192-857c85f6b5', '53f1acb37db941b8b9c77dfefecb157b-67c4bbcb66', '8b92c25a69954a66bb13dcaf394b4499-698db669f8', 
                                        'd7189bdd1e8b43b59a5f6284bb89798a-fd57867bf', '8b18231981e0440488bbac370b1464cf-68c85f66f7', '69ebe09877684907811e840ba66ea578-5969f446c5', 
                                        '883cd056805347a5b0482e2cbf1cd97f-5f5996475f', '6ddef64b548f45b49da90a040777bb4f-5484c9d98f', '0bea01d71c834b0bba9aaa41e9884cf8-58fdd89ffc', 
                                        '03d1f58da52d49dbb815cda9be061d25-c8949d9d9', '68fbe2fc0a48404c987dcb108906349a-67f9d6978c', '72d37257c88046aba2283fd7e602dfae-7cfb745bd5', 
                                        'fabe1018f1c2459a84864ddfecb30f3a-7676d54bb', '8b3eee3cc4fe4568b5ba4125c1a4047f-b57494557', '6adddd8cff25463a97ee86a9739c86d3-5d9898c9bb', 
                                        'skywalking-ui', '26490d5901ce44aabcbc33dbc5925adc-66ddf858cc', 'ad8fb26af5154e5b9dc4fb1f609282fb-5f9c4dc84c', 
                                        '63555cbf9b6341a99dd1dc5494158a28-697dddf596', 'prometheus-989b58f9', 'c453a975de5148e4b1c47be258a646c9-7999d9d4f8', 
                                        '6cc7dc7bb5fa4327a20c883ab00ab2fe-675b8fd8b', '2170e75abdf54178afcd5ffecb387eee-ddbfb8846', 'elasticsearch-58cd769777', 
                                        '848ec944cf1a4391be09dd24c105aea9-6dd6cc5cf', '192756d8271842a9a06b4252ff4e5e7b-b9d8b4895', '9ab2c6acd4ca4925955bd41e23016f5c-79c99665c8', 
                                        '36c4ac32f7504f13b7aef941de9ecc81-cb495dcfc', 'e97a387ed0204878b0660f0090bfacd6-57f599655b', 'kube-flannel', 
                                        '177495fd11344929857a237816afbb41-747cfdd897', 'faf90b12d1cf478e810172eb6aced658-64b8b7dd6f', 'alertmanager-777cf86864', 
                                        'f1023ca9976e4a5eaaaaed244acd2f4a-7f86c4c4bb', 'ea4bdf00441c4157a99a9c72bb7f4eb2-5bc6cf98b5', '05be78908e3c4818b1caa00b71d8bb11-5bcfffdd99', 
                                        '4287f5cca47742008a8fb965908e5dea-744b568489']
metric_tag_cpu7_2container_3_empty_4instance1_pod1=['kube-flannel']
metric_tag_cpu7_2container_3_notemptyct_pod57=metric_tag_cpu7_2container_3_pod57
metric_tag_cpu7_2container_3_notemptyct_4instance_pod2=['nacosdb-mysql', 'kube-flannel']
metric_tag_cpu7_2container_3_notemptyct_4instance_pod2=['kube-flannel']
metric_tag_cpu7_2container_3_notempty2ct_4instance_pod1=['nacosdb-mysql']
metric_tag_cpu7_2container_3_notempty2ct_1instance_pod3=['7f1cbe89dd024b6ebbf5556426b34acf', 'c019762f3cbd493cb4dc4443eec2c273', 'ab1c1a0046754e49b4fdf64708124365']


metric_tag_cpu7_3container_4=['container_cpu_load_average_10s','container_cpu_usage_seconds_total','container_cpu_user_seconds_total','container_cpu_system_seconds_total']
metric_tag_cpu7_3container_4_4_instance7=['kube-flannel', '', 'nacosdb-mysql','openebs-ndm', 'kube-proxy', 'node-exporter', 'chaosblade-tool']

metric_tag_net8_4instance_pod7=['','chaosblade-tool', 'openebs-ndm', 'kube-flannel', 'node-exporter', 'nacosdb-mysql', 'kube-proxy']


metric_node_jobs3=['kubernetes-pods', 'kubernetes-service-endpoints', 'node-exporter']

metric_ts=[i*60000 for i in range(21)]

metric_tags_109=list(filter(lambda x: x not in ['cpm','resp_time','error_count','success_rate'],metric_tags))
metric_tags_94=list(filter(lambda x: x not in metric_tags_15,metric_tags_109))


def metric_container_instance_str2num(x):
    if x=='node-master':
        return 4
    elif x== 'node-worker1':
        return 1
    elif x== 'node-worker2':
        return 2
    elif x== 'node-worker3':
        return 3
    else:
        return 0  

num_classes = 24

def sScore(y_true, y_pred):
    score = []
    for i in range(num_classes):
        score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        
    return score

def gen_label(train):
    col = np.zeros((train.shape[0], 24))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1
          
    return col
