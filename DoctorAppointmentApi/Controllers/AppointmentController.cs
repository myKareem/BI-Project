using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using DoctorAppointmentApi.Data;
using DoctorAppointmentApi.Models;

namespace DoctorAppointmentApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AppointmentController : ControllerBase
    {
        private readonly AppDbContext _context;

        public AppointmentController(AppDbContext context)
        {
            _context = context;
        }

        //API 2 RETRIEVE 

        [HttpGet]
        public async Task<ActionResult<IEnumerable<Appointment>>> GetAppointments()
        {
            var appointments = await _context.Appointments.ToListAsync();

            return Ok(appointments);

        }

        //API 1 Create

        [HttpPost]
        public async Task<ActionResult<Appointment>> PostAppointment([FromBody] Appointment appointment)
        {
            if (string.IsNullOrWhiteSpace(appointment.PatientName) || appointment.AppointmentDateTime == default)
            {
                return BadRequest("Patient name and a valid appointment date/time are needed.");
            }

            _context.Appointments.Add(appointment);

            await _context.SaveChangesAsync();

            return Created($"/api/appointments/{appointment.Id}", appointment);

        }
        



    }
}
