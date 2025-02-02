use Parc;

SELECT TOP (1000) [Vehicle Type]
      ,[Make]
      ,[Range]
      ,[Body Style]
      ,[Company_Private]
      ,[Number of Seats]
      ,[Colour]
      ,[Year of 1st Reg]
      ,[Count of Registrations]
  FROM [Parc].[dbo].[DataShop]
where [Number of Seats] = 2;
