create table Payment_Status (
	payment_Status_Code INT,
	payment_Status_Description TEXT,
	PRIMARY KEY(payment_Status_Code)
);

insert into Payment_Status(payment_Status_Code, payment_Status_Description) values (1, 'Pending');
insert into Payment_Status(payment_Status_Code, payment_Status_Description) values (2, 'Accepted');
insert into Payment_Status(payment_Status_Code, payment_Status_Description) values (3, 'Cancled');
insert into Payment_Status(payment_Status_Code, payment_Status_Description) values (4, 'Rejected');
insert into Payment_Status(payment_Status_Code, payment_Status_Description) values (5, 'Missing Funds');
