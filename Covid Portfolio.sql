--COVID DATA EXPORATION
--Filtering Data
SELECT location, date,population, 
total_cases, new_cases, total_deaths
FROM PortfolioProject..Cases
ORDER BY 1,2

--Total Cases vs Total Deaths
SELECT location, date,total_cases, total_deaths,
(total_deaths/total_cases)*100 AS DeathsPerCase
FROM PortfolioProject..Cases
ORDER BY 1,2

--Total Cases vs Total Deaths in Jamaica
SELECT location, date,total_cases, total_deaths,
(total_deaths/total_cases)*100 AS DeathsPerCase
FROM PortfolioProject..Cases
Where location = 'Jamaica'
ORDER BY 1,2

--Total Cases vs Population in Jamaica
SELECT location, date,population,total_cases, 
(total_cases/population)*100 AS PercentInfected
FROM PortfolioProject..Cases
WHERE location = 'Jamaica'
ORDER BY 1,2

--Countries in  North America with Highest Infection Rate
SELECT location, population,MAX(total_cases) AS InfectionCount, 
MAX((total_cases/population))*100 AS PercentInfected
FROM PortfolioProject..Cases
GROUP BY location, population
ORDER BY PercentInfected DESC

--Countries in  North America with Highest Death Rate
SELECT location, MAX(total_cases) AS InfectionCount,
MAX(total_deaths) AS DeathCount, 
(MAX(total_deaths)/MAX(total_cases))*100 AS PercentDeaths
FROM PortfolioProject..Cases
GROUP BY location
ORDER BY PercentDeaths DESC

--AGGREGATE NORTH AMERICA 
--this wrong
SELECT SUM(new_cases) AS TotalCases, 
SUM(CAST(new_deaths as int)) as TotalDeaths,
SUM(CAST(new_deaths as int))/SUM(new_cases)*100 as DeathPercentage
FROM PortfolioProject..Cases
ORDER BY 1,2


--Total Population vs Vaccinations
--CTE
WITH PopVsVacc (location, date, population, new_vaccinations, RolledVacc) 
as
(
SELECT dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(CAST(vac.new_vaccinations as int)) 
OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) 
AS RolledVacc
FROM PortfolioProject..Cases dea
JOIN PortfolioProject..CovidVaccinations vac
ON dea.location = vac.location
AND dea.date = vac.date
)
SELECT * , (RolledVacc/population)*100 AS PercentageVaccinated
FROM PopVsVacc

--TEMP TABLE
DROP TABLE IF EXISTS #PercentVaccinated
CREATE TABLE #PercentVaccinated
(
Location nvarchar(255),
Date datetime,
Population numeric,
New_Vaccinations numeric,
RolledVaccination numeric
)
INSERT INTO #PercentVaccinated
SELECT dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(CAST(vac.new_vaccinations as int)) 
OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) 
AS RolledVacc
FROM PortfolioProject..Cases dea
JOIN PortfolioProject..CovidVaccinations vac
ON dea.location = vac.location
AND dea.date = vac.date

SELECT *, (RolledVaccination/population)*100
FROM #PercentVaccinated


--View to Store Data for later use
CREATE VIEW HighestDeathRate as
--Countries in  North America with Highest Death Rate
SELECT location, MAX(total_cases) AS InfectionCount,
MAX(total_deaths) AS DeathCount, 
(MAX(total_deaths)/MAX(total_cases))*100 AS PercentDeaths
FROM PortfolioProject..Cases
GROUP BY location




