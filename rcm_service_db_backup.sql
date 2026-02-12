/*
SQLyog Community
MySQL - 5.7.44-log : Database - rcm_service
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
/*Table structure for table `imagine_records` */

CREATE TABLE `imagine_records` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `file_name` varchar(256) DEFAULT NULL,
  `file_type` varchar(16) DEFAULT NULL,
  `source` varchar(16) DEFAULT NULL,
  `file_id` varchar(256) DEFAULT NULL,
  `patient_id` varchar(256) DEFAULT NULL,
  `patient_name` varchar(512) DEFAULT NULL,
  `date_of_service` date DEFAULT NULL,
  `report` text,
  `duplicate_record` bit(1) DEFAULT NULL,
  `status` varchar(16) DEFAULT NULL,
  `log` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=latin1;

/*Data for the table `imagine_records` */

insert  into `imagine_records`(`id`,`file_name`,`file_type`,`source`,`file_id`,`patient_id`,`patient_name`,`date_of_service`,`report`,`duplicate_record`,`status`,`log`,`created_at`,`updated_at`) values 
(6,'test_file_v1.csv','CSV','BATCH','1','174780','BREAZ, BRIAN L','2025-10-27','MRN: SVH35111474\nOrder No.: 37-XR-25-048942SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 10/27/2025\nExam Description: XR CHEST 2 VIEWS\nBill Type: P\n\nReason for Exam: Abdominal Pain; Other (Please Specify)\n\nProcedure Note:\n\nCHEST - 2 VIEWS:\n\nCLINICAL HISTORY: Abdominal Pain; Other (Please Specify)\n\nComparison: 10/1/2025..\n\nFINDINGS: Two views of the chest were obtained. Heart size is stable and the vasculature unremarkable. Sternal wires again noted.\n\nMediastinal contour is normal. Elevated right hemi...',0x01,'CREATED',NULL,'2026-02-10 17:11:13','2026-02-10 17:12:44'),
(7,'test_file_v1.csv','CSV','BATCH','2','174780','BREAZ, BRIAN L','2025-11-23','MRN: SVH35111474\nOrder No.: 37-XR-25-053302SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/23/2025\nExam Description: XR SHOULDER 2 OR MORE VIEWS RIGHT\nBill Type: P\n\nReason for Exam: Pain - non trauma\n\nProcedure Note:\n\nXR SHOULDER:\n\nHISTORY: Pain - non trauma\n\nComparison: Contralateral shoulder 12/22/2020, bilateral shoulders 3/26/2025\n\nTECHNIQUE: Right shoulder, 3 views.\n\nFINDINGS: Suboptimal positioning. negative for dislocation. Cortical margins are maintained\n\nIMPRESSION:  No acute findings\n\n\nDic...',0x01,'CREATED',NULL,'2026-02-10 17:11:13','2026-02-10 17:12:44'),
(8,'test_file_v1.csv','CSV','BATCH','3','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043300CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CHEST ABDOMEN PELVIS WITH IV CONTRAST\nBill Type: P\n\nReason for Exam: Abdomen-Pelvis Trauma, Moderate, Blunt; Other (Please Specify)\n\nProcedure Note:\n\nCT CHEST ABDOMEN PELVIS WITH CONTRAST:\n\nHISTORY: Abdomen-Pelvis Trauma. Moderate. Blunt; Other (Please Specify)\n\nComparison: CTA chest 9/12/2021 and CT abdomen/pelvis 1/29/2021\n\nTECHNIQUE: After the uneventful intravenous administration...',0x01,'CREATED',NULL,'2026-02-10 17:11:13','2026-02-10 17:12:44'),
(9,'test_file_v1.csv','CSV','BATCH','4','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043300CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CHEST ABDOMEN PELVIS WITH IV CONTRAST\nBill Type: P\n\nReason for Exam: Abdomen-Pelvis Trauma, Moderate, Blunt; Other (Please Specify)\n\nProcedure Note:\n\nCT CHEST ABDOMEN PELVIS WITH CONTRAST:\n\nHISTORY: Abdomen-Pelvis Trauma. Moderate. Blunt; Other (Please Specify)\n\nComparison: CTA chest 9/12/2021 and CT abdomen/pelvis 1/29/2021\n\nTECHNIQUE: After the uneventful intravenous administration...',0x01,'CREATED',NULL,'2026-02-10 17:11:13','2026-02-10 17:12:44'),
(10,'test_file_v1.csv','CSV','BATCH','5','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043302CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CERVICAL SPINE WITHOUT IV CONTRAST\nBill Type: P\n\nReason for Exam: trauma; Other (Please Specify)\n\nProcedure Note:\n\nCT CERVICAL SPINE WITHOUT CONTRAST:\n\nHISTORY: trauma; Other (Please Specify)\n\nComparison: 10/15/2023\n\nTECHNIQUE: Thin section axial CT images were obtained from the foramen magnum to the T1 vertebral body. This CT exam was performed using one or more of the following dos...',0x01,'CREATED',NULL,'2026-02-10 17:11:13','2026-02-10 17:12:44');

/*Table structure for table `imagine_records_duplicates` */

CREATE TABLE `imagine_records_duplicates` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `file_name` varchar(256) DEFAULT NULL,
  `file_type` varchar(16) DEFAULT NULL,
  `source` varchar(16) DEFAULT NULL,
  `file_id` varchar(256) DEFAULT NULL,
  `patient_id` varchar(256) DEFAULT NULL,
  `patient_name` varchar(512) DEFAULT NULL,
  `date_of_service` date DEFAULT NULL,
  `report` text,
  `status` varchar(16) DEFAULT NULL,
  `log` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=latin1;

/*Data for the table `imagine_records_duplicates` */

insert  into `imagine_records_duplicates`(`id`,`file_name`,`file_type`,`source`,`file_id`,`patient_id`,`patient_name`,`date_of_service`,`report`,`status`,`log`,`created_at`,`updated_at`) values 
(1,'test_file_v1.csv','CSV','BATCH','1','174780','BREAZ, BRIAN L','2025-10-27','MRN: SVH35111474\nOrder No.: 37-XR-25-048942SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 10/27/2025\nExam Description: XR CHEST 2 VIEWS\nBill Type: P\n\nReason for Exam: Abdominal Pain; Other (Please Specify)\n\nProcedure Note:\n\nCHEST - 2 VIEWS:\n\nCLINICAL HISTORY: Abdominal Pain; Other (Please Specify)\n\nComparison: 10/1/2025..\n\nFINDINGS: Two views of the chest were obtained. Heart size is stable and the vasculature unremarkable. Sternal wires again noted.\n\nMediastinal contour is normal. Elevated right hemi...','CREATED',NULL,'2026-02-02 17:47:22','2026-02-02 17:47:22'),
(2,'test_file_v1.csv','CSV','BATCH','2','174780','BREAZ, BRIAN L','2025-11-23','MRN: SVH35111474\nOrder No.: 37-XR-25-053302SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/23/2025\nExam Description: XR SHOULDER 2 OR MORE VIEWS RIGHT\nBill Type: P\n\nReason for Exam: Pain - non trauma\n\nProcedure Note:\n\nXR SHOULDER:\n\nHISTORY: Pain - non trauma\n\nComparison: Contralateral shoulder 12/22/2020, bilateral shoulders 3/26/2025\n\nTECHNIQUE: Right shoulder, 3 views.\n\nFINDINGS: Suboptimal positioning. negative for dislocation. Cortical margins are maintained\n\nIMPRESSION:  No acute findings\n\n\nDic...','CREATED',NULL,'2026-02-02 17:47:22','2026-02-02 17:47:22'),
(3,'test_file_v1.csv','CSV','BATCH','3','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043300CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CHEST ABDOMEN PELVIS WITH IV CONTRAST\nBill Type: P\n\nReason for Exam: Abdomen-Pelvis Trauma, Moderate, Blunt; Other (Please Specify)\n\nProcedure Note:\n\nCT CHEST ABDOMEN PELVIS WITH CONTRAST:\n\nHISTORY: Abdomen-Pelvis Trauma. Moderate. Blunt; Other (Please Specify)\n\nComparison: CTA chest 9/12/2021 and CT abdomen/pelvis 1/29/2021\n\nTECHNIQUE: After the uneventful intravenous administration...','CREATED',NULL,'2026-02-02 17:47:22','2026-02-02 17:47:22'),
(4,'test_file_v1.csv','CSV','BATCH','4','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043300CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CHEST ABDOMEN PELVIS WITH IV CONTRAST\nBill Type: P\n\nReason for Exam: Abdomen-Pelvis Trauma, Moderate, Blunt; Other (Please Specify)\n\nProcedure Note:\n\nCT CHEST ABDOMEN PELVIS WITH CONTRAST:\n\nHISTORY: Abdomen-Pelvis Trauma. Moderate. Blunt; Other (Please Specify)\n\nComparison: CTA chest 9/12/2021 and CT abdomen/pelvis 1/29/2021\n\nTECHNIQUE: After the uneventful intravenous administration...','CREATED',NULL,'2026-02-02 17:47:22','2026-02-02 17:47:22'),
(5,'test_file_v1.csv','CSV','BATCH','5','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043302CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CERVICAL SPINE WITHOUT IV CONTRAST\nBill Type: P\n\nReason for Exam: trauma; Other (Please Specify)\n\nProcedure Note:\n\nCT CERVICAL SPINE WITHOUT CONTRAST:\n\nHISTORY: trauma; Other (Please Specify)\n\nComparison: 10/15/2023\n\nTECHNIQUE: Thin section axial CT images were obtained from the foramen magnum to the T1 vertebral body. This CT exam was performed using one or more of the following dos...','CREATED',NULL,'2026-02-02 17:47:22','2026-02-02 17:47:22'),
(6,'test_file_v1.csv','CSV','BATCH','1','174780','BREAZ, BRIAN L','2025-10-27','MRN: SVH35111474\nOrder No.: 37-XR-25-048942SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 10/27/2025\nExam Description: XR CHEST 2 VIEWS\nBill Type: P\n\nReason for Exam: Abdominal Pain; Other (Please Specify)\n\nProcedure Note:\n\nCHEST - 2 VIEWS:\n\nCLINICAL HISTORY: Abdominal Pain; Other (Please Specify)\n\nComparison: 10/1/2025..\n\nFINDINGS: Two views of the chest were obtained. Heart size is stable and the vasculature unremarkable. Sternal wires again noted.\n\nMediastinal contour is normal. Elevated right hemi...','CREATED',NULL,'2026-02-10 17:12:44','2026-02-10 17:12:44'),
(7,'test_file_v1.csv','CSV','BATCH','2','174780','BREAZ, BRIAN L','2025-11-23','MRN: SVH35111474\nOrder No.: 37-XR-25-053302SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/23/2025\nExam Description: XR SHOULDER 2 OR MORE VIEWS RIGHT\nBill Type: P\n\nReason for Exam: Pain - non trauma\n\nProcedure Note:\n\nXR SHOULDER:\n\nHISTORY: Pain - non trauma\n\nComparison: Contralateral shoulder 12/22/2020, bilateral shoulders 3/26/2025\n\nTECHNIQUE: Right shoulder, 3 views.\n\nFINDINGS: Suboptimal positioning. negative for dislocation. Cortical margins are maintained\n\nIMPRESSION:  No acute findings\n\n\nDic...','CREATED',NULL,'2026-02-10 17:12:44','2026-02-10 17:12:44'),
(8,'test_file_v1.csv','CSV','BATCH','3','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043300CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CHEST ABDOMEN PELVIS WITH IV CONTRAST\nBill Type: P\n\nReason for Exam: Abdomen-Pelvis Trauma, Moderate, Blunt; Other (Please Specify)\n\nProcedure Note:\n\nCT CHEST ABDOMEN PELVIS WITH CONTRAST:\n\nHISTORY: Abdomen-Pelvis Trauma. Moderate. Blunt; Other (Please Specify)\n\nComparison: CTA chest 9/12/2021 and CT abdomen/pelvis 1/29/2021\n\nTECHNIQUE: After the uneventful intravenous administration...','CREATED',NULL,'2026-02-10 17:12:44','2026-02-10 17:12:44'),
(9,'test_file_v1.csv','CSV','BATCH','4','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043300CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CHEST ABDOMEN PELVIS WITH IV CONTRAST\nBill Type: P\n\nReason for Exam: Abdomen-Pelvis Trauma, Moderate, Blunt; Other (Please Specify)\n\nProcedure Note:\n\nCT CHEST ABDOMEN PELVIS WITH CONTRAST:\n\nHISTORY: Abdomen-Pelvis Trauma. Moderate. Blunt; Other (Please Specify)\n\nComparison: CTA chest 9/12/2021 and CT abdomen/pelvis 1/29/2021\n\nTECHNIQUE: After the uneventful intravenous administration...','CREATED',NULL,'2026-02-10 17:12:44','2026-02-10 17:12:44'),
(10,'test_file_v1.csv','CSV','BATCH','5','175571','BERLEHNER, GARY L','2025-11-24','MRN: CHH7234022\nOrder No.: 40-CT-25-043302CHH\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/24/2025\nExam Description: CT CERVICAL SPINE WITHOUT IV CONTRAST\nBill Type: P\n\nReason for Exam: trauma; Other (Please Specify)\n\nProcedure Note:\n\nCT CERVICAL SPINE WITHOUT CONTRAST:\n\nHISTORY: trauma; Other (Please Specify)\n\nComparison: 10/15/2023\n\nTECHNIQUE: Thin section axial CT images were obtained from the foramen magnum to the T1 vertebral body. This CT exam was performed using one or more of the following dos...','CREATED',NULL,'2026-02-10 17:12:44','2026-02-10 17:12:44');

/*Table structure for table `imagine_records_results` */

CREATE TABLE `imagine_records_results` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `patient_id` varchar(256) DEFAULT NULL,
  `patient_name` varchar(512) DEFAULT NULL,
  `date_of_service` date DEFAULT NULL,
  `report` text,
  `cpt` varchar(64) DEFAULT NULL,
  `modifier` varchar(64) DEFAULT NULL,
  `icd10_diagnosis` varchar(1024) DEFAULT NULL,
  `confidence_score` decimal(6,2) DEFAULT NULL,
  `rag_match_score` decimal(6,2) DEFAULT NULL,
  `cpt_was_extracted` bit(1) DEFAULT NULL,
  `has_medical_necessity` bit(1) DEFAULT NULL,
  `medical_necessity_warning` text,
  `llm_raw_response` longtext,
  `preprocess_time_ms` int(11) DEFAULT NULL,
  `llm_time_ms` int(11) DEFAULT NULL,
  `postprocess_time_ms` int(11) DEFAULT NULL,
  `batch_id` varchar(128) DEFAULT NULL,
  `record_idx` int(11) DEFAULT NULL,
  `processed_at` timestamp(6) NULL DEFAULT NULL,
  `status` varchar(32) DEFAULT NULL,
  `source_file` varchar(512) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=44 DEFAULT CHARSET=latin1;

/*Data for the table `imagine_records_results` */

insert  into `imagine_records_results`(`id`,`patient_id`,`patient_name`,`date_of_service`,`report`,`cpt`,`modifier`,`icd10_diagnosis`,`confidence_score`,`rag_match_score`,`cpt_was_extracted`,`has_medical_necessity`,`medical_necessity_warning`,`llm_raw_response`,`preprocess_time_ms`,`llm_time_ms`,`postprocess_time_ms`,`batch_id`,`record_idx`,`processed_at`,`status`,`source_file`,`created_at`,`updated_at`) values 
(42,'174780','BREAZ, BRIAN L','2025-10-27','MRN: SVH35111474\nOrder No.: 37-XR-25-048942SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 10/27/2025\nExam Description: XR CHEST 2 VIEWS\nBill Type: P\n\nReason for Exam: Abdominal Pain; Other (Please Specify)\n\nProcedure Note:\n\nCHEST - 2 VIEWS:\n\nCLINICAL HISTORY: Abdominal Pain; Other (Please Specify)\n\nComparison: 10/1/2025..\n\nFINDINGS: Two views of the chest were obtained. Heart size is stable and the vasculature unremarkable. Sternal wires again noted.\n\nMediastinal contour is normal. Elevated right hemi...','ERROR','','',0.00,0.00,0x00,0x00,'','',0,0,0,'20260210_111727',1,'2026-02-10 11:20:14.327130','ERROR','test_file_v1.csv','2026-02-10 16:50:14','2026-02-10 16:50:14'),
(43,'174780','BREAZ, BRIAN L','2025-11-23','MRN: SVH35111474\nOrder No.: 37-XR-25-053302SVH2\nCDSM Tool:\nCDSM Outcome:\nExam Date: 11/23/2025\nExam Description: XR SHOULDER 2 OR MORE VIEWS RIGHT\nBill Type: P\n\nReason for Exam: Pain - non trauma\n\nProcedure Note:\n\nXR SHOULDER:\n\nHISTORY: Pain - non trauma\n\nComparison: Contralateral shoulder 12/22/2020, bilateral shoulders 3/26/2025\n\nTECHNIQUE: Right shoulder, 3 views.\n\nFINDINGS: Suboptimal positioning. negative for dislocation. Cortical margins are maintained\n\nIMPRESSION:  No acute findings\n\n\nDic...','ERROR','','',0.00,0.00,0x00,0x00,'','',0,0,0,'20260210_111727',2,'2026-02-10 11:20:23.020174','ERROR','test_file_v1.csv','2026-02-10 16:50:23','2026-02-10 16:50:23');

/*Table structure for table `system_liquibase_change_log` */

CREATE TABLE `system_liquibase_change_log` (
  `ID` varchar(255) NOT NULL,
  `AUTHOR` varchar(255) NOT NULL,
  `FILENAME` varchar(255) NOT NULL,
  `DATEEXECUTED` datetime NOT NULL,
  `ORDEREXECUTED` int(11) NOT NULL,
  `EXECTYPE` varchar(10) NOT NULL,
  `MD5SUM` varchar(35) DEFAULT NULL,
  `DESCRIPTION` varchar(255) DEFAULT NULL,
  `COMMENTS` varchar(255) DEFAULT NULL,
  `TAG` varchar(255) DEFAULT NULL,
  `LIQUIBASE` varchar(20) DEFAULT NULL,
  `CONTEXTS` varchar(255) DEFAULT NULL,
  `LABELS` varchar(255) DEFAULT NULL,
  `DEPLOYMENT_ID` varchar(10) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `system_liquibase_change_log` */

insert  into `system_liquibase_change_log`(`ID`,`AUTHOR`,`FILENAME`,`DATEEXECUTED`,`ORDEREXECUTED`,`EXECTYPE`,`MD5SUM`,`DESCRIPTION`,`COMMENTS`,`TAG`,`LIQUIBASE`,`CONTEXTS`,`LABELS`,`DEPLOYMENT_ID`) values 
('1','sagarp','changelogs/ver001_create_system_shedlock_table.yaml','2025-11-28 17:57:24',1,'EXECUTED','8:941586a3eac912ddf888ecc4629b6c3d','createTable tableName=system_shedlock','',NULL,'4.20.0',NULL,NULL,'4332844098'),
('1','sagarp','changelogs/ver002_create_imagine_records_table.yaml','2026-01-29 15:01:05',2,'EXECUTED','8:529afa3639c85781eab7b71ad6424ef6','createTable tableName=imagine_records','',NULL,'4.20.0',NULL,NULL,'9679065092'),
('1','sagarp','changelogs/ver003_create_imagine_records_duplicates_table.yaml','2026-02-02 13:56:30',3,'EXECUTED','8:f2e2a8cdf389be3d74a988d1094de55a','createTable tableName=imagine_records_duplicates','',NULL,'4.20.0',NULL,NULL,'0020790901'),
('1','sagarp','changelogs/ver004_create_imagine_records_results_table.yaml','2026-02-09 15:37:14',4,'EXECUTED','8:a04c31835c2894b2f04b07d25f0a4765','createTable tableName=imagine_records_results','',NULL,'4.20.0',NULL,NULL,'0631634332');

/*Table structure for table `system_liquibase_change_log_lock` */

CREATE TABLE `system_liquibase_change_log_lock` (
  `ID` int(11) NOT NULL,
  `LOCKED` bit(1) NOT NULL,
  `LOCKGRANTED` datetime DEFAULT NULL,
  `LOCKEDBY` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `system_liquibase_change_log_lock` */

insert  into `system_liquibase_change_log_lock`(`ID`,`LOCKED`,`LOCKGRANTED`,`LOCKEDBY`) values 
(1,0x00,NULL,NULL);

/*Table structure for table `system_shedlock` */

CREATE TABLE `system_shedlock` (
  `name` varchar(64) NOT NULL,
  `lock_until` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `locked_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `locked_by` varchar(255) NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `system_shedlock` */

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
